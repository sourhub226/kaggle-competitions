import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Advanced models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

print("="*70)
print("🚀 ADVANCED EXAM SCORE PREDICTION SYSTEM")
print("="*70)

# Load data
print("\n📂 Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(f"✅ Training data: {train_df.shape[0]} samples, {train_df.shape[1]} features")
print(f"✅ Test data: {test_df.shape[0]} samples")

# Advanced Feature Engineering
def advanced_feature_engineering(df):
    """Create extensive feature set with domain knowledge"""
    df = df.copy()
    
    # Basic interactions
    df['study_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
    df['attendance_study_product'] = df['class_attendance'] * df['study_hours']
    df['total_hours'] = df['study_hours'] + df['sleep_hours']
    
    # Sleep quality mapping with more granularity
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    df['sleep_quality_encoded'] = df['sleep_quality'].map(sleep_map).fillna(1)
    df['sleep_efficiency'] = df['sleep_hours'] * df['sleep_quality_encoded']
    df['effective_study_time'] = df['study_hours'] * (df['sleep_quality_encoded'] + 1) / 2
    
    # Polynomial features for key variables
    df['study_hours_sq'] = df['study_hours'] ** 2
    df['study_hours_cube'] = df['study_hours'] ** 3
    df['sleep_hours_sq'] = df['sleep_hours'] ** 2
    df['attendance_sq'] = df['class_attendance'] ** 2
    
    # Logarithmic transformations
    df['log_study_hours'] = np.log1p(df['study_hours'])
    df['log_sleep_hours'] = np.log1p(df['sleep_hours'])
    
    # Square root transformations
    df['sqrt_study_hours'] = np.sqrt(df['study_hours'])
    df['sqrt_attendance'] = np.sqrt(df['class_attendance'])
    
    # Binned features
    df['study_hours_bin'] = pd.cut(df['study_hours'], bins=5, labels=False)
    df['sleep_hours_bin'] = pd.cut(df['sleep_hours'], bins=5, labels=False)
    
    # Study method effectiveness (based on domain knowledge)
    method_effectiveness = {
        'online videos': 0.7,
        'reading': 0.8,
        'group study': 0.75,
        'practice tests': 0.9,
        'flashcards': 0.65
    }
    df['study_method_score'] = df['study_method'].map(method_effectiveness).fillna(0.7)
    df['effective_study_score'] = df['study_hours'] * df['study_method_score']
    
    # Facility rating impact
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    df['facility_encoded'] = df['facility_rating'].map(facility_map).fillna(1)
    df['facility_study_interaction'] = df['facility_encoded'] * df['study_hours']
    
    # Difficulty adjustment
    difficulty_map = {'easy': 1, 'medium': 0.85, 'hard': 0.7}
    df['difficulty_multiplier'] = df['exam_difficulty'].map(difficulty_map).fillna(0.85)
    
    # Internet access impact
    df['internet_access_encoded'] = (df['internet_access'] == 'yes').astype(int)
    df['internet_study_interaction'] = df['internet_access_encoded'] * df['study_hours']
    
    # Age-based features
    df['age_study_ratio'] = df['age'] / (df['study_hours'] + 1)
    df['age_normalized'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # Cross-feature interactions
    df['attendance_sleep_product'] = df['class_attendance'] * df['sleep_hours']
    df['study_attendance_ratio'] = df['study_hours'] / (df['class_attendance'] + 1)
    
    # Performance indicators
    df['overall_dedication'] = (
        (df['study_hours'] / df['study_hours'].max()) * 0.4 +
        (df['class_attendance'] / 100) * 0.3 +
        (df['sleep_quality_encoded'] / 2) * 0.3
    )
    
    return df

# Apply feature engineering
print("\n🔧 Engineering features...")
X = train_df.drop(['exam_score'], axis=1)
y = train_df['exam_score']
test_ids = test_df['id']

X = advanced_feature_engineering(X)
test_df_processed = advanced_feature_engineering(test_df)

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X_processed = X.copy()
test_processed = test_df_processed.copy()

label_encoders = {}
for col in tqdm(categorical_cols, desc="Encoding categories"):
    le = LabelEncoder()
    combined = pd.concat([X_processed[col], test_processed[col]], axis=0)
    le.fit(combined)
    X_processed[col] = le.transform(X_processed[col])
    test_processed[col] = le.transform(test_processed[col])
    label_encoders[col] = le

# Remove id
X_processed = X_processed.drop(['id'], axis=1)
test_processed = test_processed.drop(['id'], axis=1)

# Robust scaling (handles outliers better)
print("\n📊 Scaling features with RobustScaler...")
scaler = RobustScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_processed),
    columns=X_processed.columns,
    index=X_processed.index
)
test_scaled = pd.DataFrame(
    scaler.transform(test_processed),
    columns=test_processed.columns,
    index=test_processed.index
)

# Split with stratification approximation
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Validation set: {X_val.shape[0]} samples")
print(f"✅ Total features: {X_train.shape[1]}")

# Define optimized models
print("\n" + "="*70)
print("🎯 TRAINING ENSEMBLE OF OPTIMIZED MODELS")
print("="*70)

models = []
model_names = []
val_predictions = []

# Model 1: Optimized Random Forest
print("\n🌲 Training Optimized Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
rf_r2 = r2_score(y_val, rf_pred)
print(f"✅ Random Forest - RMSE: {rf_rmse:.3f}, R²: {rf_r2:.5f}")
models.append(rf_model)
model_names.append('Random Forest')
val_predictions.append(rf_pred)

# Model 2: Gradient Boosting with tuned parameters
print("\n📈 Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    min_samples_split=3,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
    verbose=1
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
gb_r2 = r2_score(y_val, gb_pred)
print(f"✅ Gradient Boosting - RMSE: {gb_rmse:.3f}, R²: {gb_r2:.5f}")
models.append(gb_model)
model_names.append('Gradient Boosting')
val_predictions.append(gb_pred)

# Model 3: Extra Trees
print("\n🌳 Training Extra Trees...")
et_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_val)
et_rmse = np.sqrt(mean_squared_error(y_val, et_pred))
et_r2 = r2_score(y_val, et_pred)
print(f"✅ Extra Trees - RMSE: {et_rmse:.3f}, R²: {et_r2:.5f}")
models.append(et_model)
model_names.append('Extra Trees')
val_predictions.append(et_pred)

# Model 4: XGBoost
if XGBOOST_AVAILABLE:
    print("\n🚀 Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        verbosity=2
    )
    xgb_model.fit(X_train, y_train, verbose=True)
    xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    xgb_r2 = r2_score(y_val, xgb_pred)
    print(f"✅ XGBoost - RMSE: {xgb_rmse:.3f}, R²: {xgb_r2:.5f}")
    models.append(xgb_model)
    model_names.append('XGBoost')
    val_predictions.append(xgb_pred)

# Model 5: LightGBM
if LIGHTGBM_AVAILABLE:
    print("\n💡 Training LightGBM...")
    lgbm_model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_val)
    lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_pred))
    lgbm_r2 = r2_score(y_val, lgbm_pred)
    print(f"✅ LightGBM - RMSE: {lgbm_rmse:.3f}, R²: {lgbm_r2:.5f}")
    models.append(lgbm_model)
    model_names.append('LightGBM')
    val_predictions.append(lgbm_pred)

# Model 6: CatBoost
if CATBOOST_AVAILABLE:
    print("\n🐱 Training CatBoost...")
    cat_model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=1,
        random_state=42,
        verbose=20
    )
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_val)
    cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred))
    cat_r2 = r2_score(y_val, cat_pred)
    print(f"✅ CatBoost - RMSE: {cat_rmse:.3f}, R²: {cat_r2:.5f}")
    models.append(cat_model)
    model_names.append('CatBoost')
    val_predictions.append(cat_pred)

# Model 7: Ridge Regression
print("\n📐 Training Ridge Regression...")
ridge_model = Ridge(alpha=10, random_state=42)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_val)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
ridge_r2 = r2_score(y_val, ridge_pred)
print(f"✅ Ridge - RMSE: {ridge_rmse:.3f}, R²: {ridge_r2:.5f}")
models.append(ridge_model)
model_names.append('Ridge')
val_predictions.append(ridge_pred)

# Model 8: ElasticNet
print("\n🔗 Training ElasticNet...")
elastic_model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=5000)
elastic_model.fit(X_train, y_train)
elastic_pred = elastic_model.predict(X_val)
elastic_rmse = np.sqrt(mean_squared_error(y_val, elastic_pred))
elastic_r2 = r2_score(y_val, elastic_pred)
print(f"✅ ElasticNet - RMSE: {elastic_rmse:.3f}, R²: {elastic_r2:.5f}")
models.append(elastic_model)
model_names.append('ElasticNet')
val_predictions.append(elastic_pred)

# ADVANCED: Stacking Ensemble with Meta-Learner
print("\n" + "="*70)
print("🎯 BUILDING STACKED ENSEMBLE WITH META-LEARNER")
print("="*70)

# Create base estimators for stacking
base_estimators = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42))
]

if XGBOOST_AVAILABLE:
    base_estimators.append(('xgb', XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1)))

if LIGHTGBM_AVAILABLE:
    base_estimators.append(('lgbm', LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)))

# Meta-learner
print("\n🧠 Training Stacking Regressor with Ridge meta-learner...")
stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=Ridge(alpha=1),
    cv=5,
    n_jobs=-1
)
stacking_model.fit(X_train, y_train)
stacking_pred = stacking_model.predict(X_val)
stacking_rmse = np.sqrt(mean_squared_error(y_val, stacking_pred))
stacking_r2 = r2_score(y_val, stacking_pred)
print(f"✅ Stacking Ensemble - RMSE: {stacking_rmse:.3f}, R²: {stacking_r2:.5f}")
models.append(stacking_model)
model_names.append('Stacking')
val_predictions.append(stacking_pred)

# Weighted Ensemble (give more weight to better models)
print("\n" + "="*70)
print("⚖️  CREATING OPTIMIZED WEIGHTED ENSEMBLE")
print("="*70)

# Calculate weights based on R² scores
r2_scores = []
for pred in val_predictions:
    r2_scores.append(r2_score(y_val, pred))

# Convert to weights (higher R² = higher weight)
weights = np.array(r2_scores)
weights = np.maximum(weights, 0)  # Ensure non-negative
weights = weights / weights.sum()  # Normalize

print("\n📊 Model Weights:")
for name, weight, r2 in zip(model_names, weights, r2_scores):
    print(f"   {name:20s}: {weight:.4f} (R²: {r2:.5f})")

# Weighted ensemble prediction
weighted_ensemble_pred = np.zeros(len(y_val))
for pred, weight in zip(val_predictions, weights):
    weighted_ensemble_pred += pred * weight

we_rmse = np.sqrt(mean_squared_error(y_val, weighted_ensemble_pred))
we_r2 = r2_score(y_val, weighted_ensemble_pred)
we_mae = mean_absolute_error(y_val, weighted_ensemble_pred)

print("\n" + "="*70)
print("🏆 FINAL WEIGHTED ENSEMBLE PERFORMANCE")
print("="*70)
print(f"RMSE: {we_rmse:.3f}")
print(f"R² Score: {we_r2:.5f}")
print(f"MAE: {we_mae:.3f}")
print("="*70)

# Generate test predictions
print("\n🔮 Generating predictions on test data...")
test_predictions = []
for model in tqdm(models, desc="Model predictions"):
    test_predictions.append(model.predict(test_scaled))

# Weighted ensemble on test set
final_predictions = np.zeros(len(test_scaled))
for pred, weight in zip(test_predictions, weights):
    final_predictions += pred * weight

# Save predictions
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': final_predictions
})

submission.to_csv('pred_9model.csv', index=False)

print("\n" + "="*70)
print("✅ PREDICTIONS SAVED TO 'pred_9model.csv'")
print("="*70)
print(f"\n📊 Prediction Statistics:")
print(f"   Min:    {final_predictions.min():.2f}")
print(f"   Max:    {final_predictions.max():.2f}")
print(f"   Mean:   {final_predictions.mean():.2f}")
print(f"   Median: {np.median(final_predictions):.2f}")
print(f"   Std:    {final_predictions.std():.2f}")
print(f"\n🔍 First 10 predictions:")
print(submission.head(10))
print("\n" + "="*70)
print("🎉 PREDICTION COMPLETE!")
print("="*70)