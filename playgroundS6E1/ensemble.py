import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False


# Load data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Feature Engineering
def create_features(df):
    df = df.copy()
    
    # Interaction features
    df['study_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
    df['attendance_study_product'] = df['class_attendance'] * df['study_hours']
    df['total_hours'] = df['study_hours'] + df['sleep_hours']
    df['sleep_efficiency'] = df['sleep_hours'] * df['sleep_quality'].map({
        'poor': 0.5, 'average': 1.0, 'good': 1.5
    }).fillna(1.0)
    
    # Polynomial features for important variables
    df['study_hours_sq'] = df['study_hours'] ** 2
    df['sleep_hours_sq'] = df['sleep_hours'] ** 2
    df['attendance_sq'] = df['class_attendance'] ** 2
    
    return df

# Separate features and target
X = train_df.drop(['exam_score'], axis=1)
y = train_df['exam_score']
test_ids = test_df['id']

# Apply feature engineering
X = create_features(X)
test_df_processed = create_features(test_df)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_cols:
    numerical_cols.remove('id')

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
X_processed = X.copy()
test_processed = test_df_processed.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_processed[col], test_processed[col]], axis=0)
    le.fit(combined)
    X_processed[col] = le.transform(X_processed[col])
    test_processed[col] = le.transform(test_processed[col])
    label_encoders[col] = le

# Remove id column
X_processed = X_processed.drop(['id'], axis=1)
test_processed = test_processed.drop(['id'], axis=1)

# Scale numerical features
scaler = StandardScaler()
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
test_processed[numerical_cols] = scaler.transform(test_processed[numerical_cols])

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("\nTraining multiple models...\n")

# Model 1: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
rf_r2 = r2_score(y_val, rf_pred)
print(f"Random Forest - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")

# Model 2: Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=3,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
gb_r2 = r2_score(y_val, gb_pred)
print(f"Gradient Boosting - RMSE: {gb_rmse:.2f}, R²: {gb_r2:.4f}")

# Model 3: Extra Trees
et_model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_val)
et_rmse = np.sqrt(mean_squared_error(y_val, et_pred))
et_r2 = r2_score(y_val, et_pred)
print(f"Extra Trees - RMSE: {et_rmse:.2f}, R²: {et_r2:.4f}")

# Model 4: XGBoost (if available)
models = [rf_model, gb_model, et_model]
val_preds = [rf_pred, gb_pred, et_pred]
test_preds = []

if XGBOOST_AVAILABLE:
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    xgb_r2 = r2_score(y_val, xgb_pred)
    print(f"XGBoost - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")
    models.append(xgb_model)
    val_preds.append(xgb_pred)

# Model 5: LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    lgbm_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_val)
    lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_pred))
    lgbm_r2 = r2_score(y_val, lgbm_pred)
    print(f"LightGBM - RMSE: {lgbm_rmse:.2f}, R²: {lgbm_r2:.4f}")
    models.append(lgbm_model)
    val_preds.append(lgbm_pred)

# Model 6: CatBoost (if available)
if CATBOOST_AVAILABLE:
    cat_model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_val)
    cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred))
    cat_r2 = r2_score(y_val, cat_pred)
    print(f"CatBoost - RMSE: {cat_rmse:.2f}, R²: {cat_r2:.4f}")
    models.append(cat_model)
    val_preds.append(cat_pred)

# Ensemble: Weighted average of all models
print("\n" + "="*50)
print("Creating Ensemble Model...")
ensemble_pred = np.mean(val_preds, axis=0)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
ensemble_r2 = r2_score(y_val, ensemble_pred)
print(f"Ensemble Model - RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.4f}")
print("="*50)

# Make predictions on test set
print("\nGenerating predictions on test data...")
for model in models:
    test_preds.append(model.predict(test_processed))

# Final ensemble prediction
final_predictions = np.mean(test_preds, axis=0)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': final_predictions
})

submission.to_csv('pred_ensemble.csv', index=False)
print("\nPredictions saved to 'pred_ensemble.csv'")
print(f"\nPrediction Statistics:")
print(f"Min: {final_predictions.min():.2f}")
print(f"Max: {final_predictions.max():.2f}")
print(f"Mean: {final_predictions.mean():.2f}")
print(f"Std: {final_predictions.std():.2f}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))