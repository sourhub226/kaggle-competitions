import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

print("="*70)
print("🎯 ADVANCED CSV PREDICTIONS BLENDER")
print("="*70)

# Configuration
CSV_FILES = [
    'pred_0.csv',
    'pred_9model.csv',
    'pred_ensemble.csv',
    'pred_lightgbm_catboost_xgboost.csv',
    # Add more files as needed
]

# Alternative: Auto-detect all prediction CSV files in current directory
AUTO_DETECT = False

if AUTO_DETECT:
    print("\n🔍 Auto-detecting prediction CSV files...")
    CSV_FILES = glob.glob('predictions*.csv') + glob.glob('pred*.csv') + glob.glob('submission*.csv')
    CSV_FILES = [f for f in CSV_FILES if 'blended' not in f.lower()]  # Exclude previous blended files
    
if not CSV_FILES:
    print("❌ No CSV files found!")
    print("\n💡 Please ensure your prediction files are named like:")
    print("   - predictions1.csv, predictions2.csv, etc.")
    print("   - pred1.csv, pred2.csv, etc.")
    print("   - submission1.csv, submission2.csv, etc.")
    exit(1)

print(f"\n📂 Found {len(CSV_FILES)} prediction files:")
for i, file in enumerate(CSV_FILES, 1):
    if os.path.exists(file):
        print(f"   ✅ {i}. {file}")
    else:
        print(f"   ❌ {i}. {file} (NOT FOUND)")

# Load all prediction files
print("\n📥 Loading prediction files...")
dataframes = []
valid_files = []

for file in tqdm(CSV_FILES, desc="Loading files"):
    try:
        if not os.path.exists(file):
            print(f"⚠️  Skipping {file} - file not found")
            continue
            
        df = pd.read_csv(file)
        
        # Validate format
        if 'id' not in df.columns:
            print(f"⚠️  Skipping {file} - missing 'id' column")
            continue
        
        # Find the score column (could be exam_score, score, prediction, etc.)
        score_col = None
        for col in df.columns:
            if col.lower() in ['exam_score', 'score', 'prediction', 'pred', 'target']:
                score_col = col
                break
        
        if score_col is None and len(df.columns) == 2:
            # Assume second column is the score
            score_col = df.columns[1]
        
        if score_col is None:
            print(f"⚠️  Skipping {file} - couldn't identify score column")
            continue
        
        # Rename to standard format
        df = df.rename(columns={score_col: 'exam_score'})
        
        # Ensure numeric predictions
        df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
        
        # Check for missing values
        if df['exam_score'].isna().any():
            print(f"⚠️  Warning: {file} has {df['exam_score'].isna().sum()} missing values (will be filled with mean)")
            df['exam_score'].fillna(df['exam_score'].mean(), inplace=True)
        
        dataframes.append(df[['id', 'exam_score']])
        valid_files.append(file)
        print(f"   ✅ Loaded {file}: {len(df)} rows, mean score: {df['exam_score'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error loading {file}: {str(e)}")

if len(dataframes) < 2:
    print("\n❌ Need at least 2 valid prediction files to blend!")
    exit(1)

print(f"\n✅ Successfully loaded {len(dataframes)} files for blending")

# Verify all files have same IDs
print("\n🔍 Validating IDs across files...")
base_ids = set(dataframes[0]['id'])
for i, df in enumerate(dataframes[1:], 2):
    if set(df['id']) != base_ids:
        print(f"⚠️  Warning: File {i} has different IDs than file 1")
        print(f"   This may cause issues. Proceeding with intersection of IDs...")

# Merge all predictions
print("\n🔄 Merging predictions...")
merged = dataframes[0].copy()
merged = merged.rename(columns={'exam_score': 'pred_1'})

for i, df in enumerate(dataframes[1:], 2):
    df_renamed = df.rename(columns={'exam_score': f'pred_{i}'})
    merged = merged.merge(df_renamed, on='id', how='inner')

print(f"✅ Merged {len(merged)} rows with {len(dataframes)} predictions each")

# Extract prediction columns
pred_cols = [col for col in merged.columns if col.startswith('pred_')]
predictions_array = merged[pred_cols].values

print("\n📊 Prediction Statistics per Model:")
print("-" * 70)
for i, col in enumerate(pred_cols, 1):
    stats = merged[col].describe()
    print(f"Model {i}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
          f"mean={stats['mean']:.2f}, std={stats['std']:.2f}")

# Calculate correlation between predictions
print("\n🔗 Correlation between predictions:")
correlation_matrix = merged[pred_cols].corr()
print(correlation_matrix.round(3))

# Different blending strategies
print("\n" + "="*70)
print("🎯 APPLYING BLENDING STRATEGIES")
print("="*70)

blending_results = {}

# 1. Simple Average (Equal weights)
print("\n1️⃣  Simple Average (Equal Weights)")
simple_avg = predictions_array.mean(axis=1)
blending_results['simple_average'] = simple_avg
print(f"   Mean: {simple_avg.mean():.4f}, Std: {simple_avg.std():.4f}")

# 2. Weighted Average (based on inverse variance)
print("\n2️⃣  Weighted Average (Inverse Variance)")
variances = predictions_array.var(axis=0)
weights_var = 1 / (variances + 1e-6)
weights_var = weights_var / weights_var.sum()
weighted_avg = np.average(predictions_array, axis=1, weights=weights_var)
blending_results['weighted_average'] = weighted_avg
print(f"   Weights: {weights_var.round(4)}")
print(f"   Mean: {weighted_avg.mean():.4f}, Std: {weighted_avg.std():.4f}")

# 3. Median (Robust to outliers)
print("\n3️⃣  Median (Robust to Outliers)")
median_blend = np.median(predictions_array, axis=1)
blending_results['median'] = median_blend
print(f"   Mean: {median_blend.mean():.4f}, Std: {median_blend.std():.4f}")

# 4. Trimmed Mean (Remove extremes)
print("\n4️⃣  Trimmed Mean (Remove 20% extremes)")
trimmed_mean = np.apply_along_axis(
    lambda x: np.mean(np.sort(x)[1:-1]) if len(x) > 2 else np.mean(x),
    axis=1,
    arr=predictions_array
)
blending_results['trimmed_mean'] = trimmed_mean
print(f"   Mean: {trimmed_mean.mean():.4f}, Std: {trimmed_mean.std():.4f}")

# 5. Geometric Mean
print("\n5️⃣  Geometric Mean")
geometric_mean = np.exp(np.log(predictions_array + 1e-6).mean(axis=1))
blending_results['geometric_mean'] = geometric_mean
print(f"   Mean: {geometric_mean.mean():.4f}, Std: {geometric_mean.std():.4f}")

# 6. Rank Average (Ensemble of ranks)
print("\n6️⃣  Rank Average")
from scipy.stats import rankdata
rank_predictions = np.zeros_like(predictions_array)
for i in range(predictions_array.shape[1]):
    rank_predictions[:, i] = rankdata(predictions_array[:, i])
rank_avg = rank_predictions.mean(axis=1)
# Convert ranks back to scores (normalize)
rank_avg_normalized = (rank_avg - rank_avg.min()) / (rank_avg.max() - rank_avg.min())
rank_avg_scaled = rank_avg_normalized * (predictions_array.max() - predictions_array.min()) + predictions_array.min()
blending_results['rank_average'] = rank_avg_scaled
print(f"   Mean: {rank_avg_scaled.mean():.4f}, Std: {rank_avg_scaled.std():.4f}")

# 7. Optimized Weighted Blend (power ensemble)
print("\n7️⃣  Power Ensemble (Optimized Weights)")
# Use correlation-based weights
avg_corr = correlation_matrix.mean(axis=1).values
power_weights = (1 - avg_corr + 0.1) ** 2  # Higher weight to more diverse models
power_weights = power_weights / power_weights.sum()
power_blend = np.average(predictions_array, axis=1, weights=power_weights)
blending_results['power_ensemble'] = power_blend
print(f"   Weights: {power_weights.round(4)}")
print(f"   Mean: {power_blend.mean():.4f}, Std: {power_blend.std():.4f}")

# Save all blended results
print("\n" + "="*70)
print("💾 SAVING BLENDED PREDICTIONS")
print("="*70)

output_dir = "blended_predictions"
os.makedirs(output_dir, exist_ok=True)

for name, predictions in blending_results.items():
    output_df = pd.DataFrame({
        'id': merged['id'].astype(int),
        'exam_score': predictions.astype(float).round(6)  # Ensure numeric with decimals
    })
    
    output_file = f"{output_dir}/{name}.csv"
    output_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"✅ Saved: {output_file}")

# Create a recommended blend (average of best methods)
print("\n🏆 Creating RECOMMENDED BLEND (Best Methods Combined)...")
recommended_methods = ['simple_average', 'weighted_average', 'median', 'power_ensemble']
recommended_predictions = np.column_stack([blending_results[m] for m in recommended_methods])
recommended_blend = recommended_predictions.mean(axis=1)

recommended_df = pd.DataFrame({
    'id': merged['id'].astype(int),
    'exam_score': recommended_blend.astype(float).round(6)
})
recommended_df.to_csv('predictions_RECOMMENDED.csv', index=False, float_format='%.6f')

print("\n" + "="*70)
print("🎉 BLENDING COMPLETE!")
print("="*70)
print(f"\n📊 RECOMMENDED BLEND Statistics:")
print(f"   Min:    {recommended_blend.min():.4f}")
print(f"   Max:    {recommended_blend.max():.4f}")
print(f"   Mean:   {recommended_blend.mean():.4f}")
print(f"   Median: {np.median(recommended_blend):.4f}")
print(f"   Std:    {recommended_blend.std():.4f}")

print(f"\n🎯 Best file to use: predictions_RECOMMENDED.csv")
print(f"\n💡 All blended files saved in: {output_dir}/")
print("\nAvailable blends:")
for i, name in enumerate(blending_results.keys(), 1):
    print(f"   {i}. {name}.csv")

print("\n" + "="*70)
print("📝 USAGE TIP:")
print("   Try submitting 'predictions_RECOMMENDED.csv' first!")
print("   If you want to experiment, try other blends from the folder.")
print("="*70)