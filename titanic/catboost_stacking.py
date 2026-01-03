# Cell 0: Install dependencies if not already installed
# Run this if you need to install packages. Comment out if already present.
# !pip install lightgbm catboost scikit-learn optuna --quiet

# Cell 1: Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier
import gc

# Cell 2: Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Train:", train.shape, "Test:", test.shape)

# Cell 3: Robust feature engineering function
def feature_engineering(df):
    df = df.copy()
    # Basic fills
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare','Capt':'Rare','Col':'Rare',
        'Don':'Rare','Dr':'Rare','Major':'Rare','Rev':'Rare','Sir':'Rare',
        'Jonkheer':'Rare','Dona':'Rare'
    })
    # Map small rare titles to Rare
    # Age: keep as is but add missing indicator
    df['AgeNull'] = df['Age'].isnull().astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    # Ticket group: extract prefix or length based grouping
    df['TicketPrefix'] = df['Ticket'].str.replace(r'[^A-Za-z]', ' ', regex=True).str.split().str[0].fillna('NONE')
    df['TicketPrefix'] = df['TicketPrefix'].where(df['TicketPrefix'].str.len() > 0, 'NONE')
    # Cabin deck first letter
    df['Deck'] = df['Cabin'].fillna('Unknown').str[0]
    # Drop columns we do not want to explode
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    return df

train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# Cell 4: Prepare feature lists and label encoding for categorical features used by CatBoost and LightGBM
target = 'Survived'
features = [c for c in train_fe.columns if c != target]

# Label encode low cardinality categorical features
cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)], axis=0)
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))

X = train_fe[features].copy()
y = train_fe[target].copy()
X_test = test_fe[features].copy()

print("Feature count:", len(features))

# Cell 5: Utility to create OOF preds for LightGBM and CatBoost
def get_oof_preds(models_config, X, y, X_test, n_splits=5, random_seeds=[42]):
    """
    models_config: list of dicts each containing:
       {
         'name': str,
         'type': 'lgb' or 'cat',
         'params': {...}
       }
    returns:
      oof_train_matrix: (n_train, n_models * len(seeds))  OOF predictions
      test_pred_matrix: (n_test, n_models * len(seeds)) averaged test preds per fold per seed
      model_names: list of column names
      cv_scores: dict with model -> list of fold accuracies aggregated across seeds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_models_total = len(models_config) * len(random_seeds)
    oof_train = np.zeros((X.shape[0], n_models_total))
    oof_test = np.zeros((X_test.shape[0], n_models_total))
    model_names = []
    cv_scores = {}

    col_idx = 0
    for mc in models_config:
        name = mc['name']
        cv_scores[name] = []
        for seed in random_seeds:
            fold_idx = 0
            fold_test_preds = np.zeros((X_test.shape[0], n_splits))
            fold_oof = np.zeros(X.shape[0])
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

                if mc['type'] == 'lgb':
                    train_set = lgb.Dataset(X_tr, label=y_tr)
                    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
                    params = mc['params'].copy()
                    params['seed'] = seed
                    bst = lgb.train(
                        params,
                        train_set,
                        num_boost_round=params.pop('num_boost_round', 1000),
                        valid_sets=[valid_set],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50, verbose=False)
                        ]
                    )
                    val_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
                    test_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

                elif mc['type'] == 'cat':
                    params = mc['params'].copy()
                    params['random_seed'] = seed
                    model = CatBoostClassifier(**params)
                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=50)
                    val_pred = model.predict_proba(X_val)[:, 1]
                    test_pred = model.predict_proba(X_test)[:, 1]

                else:
                    raise ValueError("Unknown model type")

                # store fold predictions
                fold_oof[val_idx] = val_pred
                fold_test_preds[:, fold] = test_pred

                # fold accuracy printed
                fold_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
                print(f"{name} seed{seed} fold{fold+1} accuracy: {fold_acc:.4f} ({fold_acc*100:.2f}%)")
                cv_scores[name].append(fold_acc)
                fold_idx += 1
                gc.collect()

            # after folds, average test preds across folds for this seed
            avg_test_pred = fold_test_preds.mean(axis=1)
            # place in column
            oof_train[:, col_idx] = fold_oof
            oof_test[:, col_idx] = avg_test_pred
            model_names.append(f"{name}_s{seed}")
            col_idx += 1

            seed_mean = np.mean(cv_scores[name][-n_splits:])  # last n_splits entries correspond to this seed
            print(f"{name} seed{seed} CV mean acc: {seed_mean:.4f} ({seed_mean*100:.2f}%)")
        print(f"Overall {name} CV mean acc (all seeds): {np.mean(cv_scores[name]):.4f} ({np.mean(cv_scores[name])*100:.2f}%)\n")

    overall_cols = model_names
    return oof_train, oof_test, overall_cols, cv_scores

# Cell 6: Define models config (tuned reasonable params)
models_config = [
    {
        'name': 'lgb_1',
        'type': 'lgb',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.02,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_freq': 5,
            'bagging_fraction': 0.8,
            'verbosity': -1,
            'num_boost_round': 1000
        }
    },
    {
        'name': 'cat_1',
        'type': 'cat',
        'params': {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'Logloss',
            'verbose': False
        }
    }
]

# Cell 7: Run OOF generation (2 seeds to reduce variance)
seeds = [42, 7]
oof_train, oof_test, model_cols, cv_scores = get_oof_preds(models_config, X, y, X_test, n_splits=5, random_seeds=seeds)

print("OOF train shape:", oof_train.shape, "OOF test shape:", oof_test.shape)

# Cell 8: Meta learner training using OOF predictions
# Print per-model mean CV accuracy
for m in cv_scores:
    print(f"{m} mean CV acc: {np.mean(cv_scores[m]):.4f} ({np.mean(cv_scores[m])*100:.2f}%)")

# Train logistic regression on OOF preds
meta = LogisticRegression(max_iter=1000)
# Evaluate meta with StratifiedKFold on OOF to estimate final stacking CV accuracy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof = np.zeros(oof_train.shape[0])
meta_cv_scores = []
for train_idx, val_idx in skf.split(oof_train, y):
    meta.fit(oof_train[train_idx], y.iloc[train_idx])
    val_pred = meta.predict(oof_train[val_idx])
    acc = accuracy_score(y.iloc[val_idx], val_pred)
    meta_cv_scores.append(acc)
    print(f"Meta fold accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("Meta CV mean accuracy:", np.mean(meta_cv_scores), f"({np.mean(meta_cv_scores)*100:.2f}%)")

# Final meta fit on full OOF
meta.fit(oof_train, y)

# Cell 9: Final prediction on test set via stacking
test_meta_pred_proba = meta.predict_proba(oof_test)[:, 1]
# threshold tuning on OOF to pick best threshold
best_thresh = 0.5
best_acc = 0.0
for t in np.linspace(0.4, 0.6, 21):
    pred = (meta.predict_proba(oof_train)[:, 1] >= t).astype(int)
    acc = accuracy_score(y, pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t
print(f"Chosen threshold on OOF: {best_thresh:.3f} yielding OOF acc: {best_acc:.4f} ({best_acc*100:.2f}%)")

final_preds = (test_meta_pred_proba >= best_thresh).astype(int)

# Cell 10: Save submission
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": final_preds.astype(int)
})
submission.to_csv("submission_stack.csv", index=False)
print("Saved submission_stack.csv with rows:", submission.shape[0])
