import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data columns:", train_df.columns.tolist())
print("\nFirst few rows of training data:")
print(train_df.head())

# Separate features and target
X = train_df.drop(['exam_score'], axis=1)
y = train_df['exam_score']

# Identify categorical columns (excluding 'id')
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", categorical_cols)

# Create a copy for processing
X_processed = X.copy()
test_processed = test_df.copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined data to ensure all categories are captured
    combined = pd.concat([X_processed[col], test_processed[col]], axis=0)
    le.fit(combined)
    
    X_processed[col] = le.transform(X_processed[col])
    test_processed[col] = le.transform(test_processed[col])
    label_encoders[col] = le

print("\nProcessed training data:")
print(X_processed.head())

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Train the model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred_val = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred_val)

print(f"\nValidation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Make predictions on test set
print("\nMaking predictions on test data...")
predictions = model.predict(test_processed)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'exam_score': predictions
})

# Save to CSV
submission.to_csv('pred_0.csv', index=False)
print("\nPredictions saved to 'pred_0.csv'")
print(f"\nFirst few predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(f"Min: {predictions.min():.2f}")
print(f"Max: {predictions.max():.2f}")
print(f"Mean: {predictions.mean():.2f}")
print(f"Median: {np.median(predictions):.2f}")