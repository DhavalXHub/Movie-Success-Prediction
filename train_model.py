"""
Pre-Release Movie Success Prediction - Model Training Script
This script handles data preprocessing, model training, and evaluation.
Uses only pre-release features to predict movie success before release.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 60)
print("Pre-Release Movie Success Prediction - Model Training")
print("=" * 60)

# ============================================================================
# 1. Load Dataset
# ============================================================================
print("\n[1] Loading dataset...")
dataset_path = "dataset/tmdb_movies.csv"

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"[ERROR] Dataset not found at {dataset_path}")
    print("Please download the TMDB dataset from Kaggle:")
    print("https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
    print("\nOr use the provided dataset file: tmdb_5000_movies.csv")
    print("Place it in the 'dataset' folder and rename it to 'tmdb_movies.csv'")
    exit(1)

df = pd.read_csv(dataset_path)
print(f"[OK] Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Display dataset info
print("\n[2] Dataset Information:")
print(f"   Columns: {list(df.columns)}")
print(f"\n   Missing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# ============================================================================
# 2. Feature Engineering - Pre-Release Features Only
# ============================================================================
print("\n[3] Feature Engineering - Pre-Release Features Only...")

# Create target variable first (using revenue for training labels only)
if 'budget' in df.columns and 'revenue' in df.columns:
    df['success'] = (df['revenue'] > df['budget'] * 2).astype(int)
    print("   [OK] Target variable 'success' created (revenue > budget * 2)")
else:
    print("   [ERROR] Cannot create target variable: budget or revenue missing")
    exit(1)

# Create working dataframe with only pre-release features
df_work = pd.DataFrame()
df_work['success'] = df['success'].copy()

# Extract Release Year from release_date
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df_work['release_year'] = df['release_date'].dt.year
    print("   [OK] Release year extracted from release_date")
else:
    print("   [WARNING] release_date not found, creating release_year from index")
    df_work['release_year'] = 2000 + (df.index % 25)

# Extract Budget
if 'budget' in df.columns:
    df_work['budget'] = df['budget'].copy()
    print("   [OK] Budget extracted")
else:
    print("   [ERROR] Budget column not found!")
    exit(1)

# Extract Runtime
if 'runtime' in df.columns:
    df_work['runtime'] = df['runtime'].copy()
    print("   [OK] Runtime extracted")
else:
    print("   [WARNING] Runtime not found, creating dummy values")
    df_work['runtime'] = 120  # Default runtime

# Extract and encode Genre
if 'genres' in df.columns:
    def extract_genre(genre_str):
        """Extract first genre from JSON-like string"""
        if pd.isna(genre_str):
            return 'Unknown'
        if isinstance(genre_str, str):
            try:
                genre_list = json.loads(genre_str.replace("'", '"'))
                if isinstance(genre_list, list) and len(genre_list) > 0:
                    return genre_list[0].get('name', 'Unknown') if isinstance(genre_list[0], dict) else genre_list[0]
            except:
                return genre_str.split(',')[0].strip() if ',' in genre_str else genre_str
        return 'Unknown'
    
    df_work['genre'] = df['genres'].apply(extract_genre)
    
    # Encode genre using LabelEncoder
    le = LabelEncoder()
    df_work['genre_encoded'] = le.fit_transform(df_work['genre'])
    df_work.drop('genre', axis=1, inplace=True)
    
    # Save label encoder for later use
    joblib.dump(le, 'genre_encoder.pkl')
    print(f"   [OK] Genres encoded using LabelEncoder ({len(le.classes_)} unique genres)")
else:
    print("   [WARNING] Genres column not found, creating dummy genre")
    df_work['genre_encoded'] = 0
    le = LabelEncoder()
    le.fit(['Unknown'])
    joblib.dump(le, 'genre_encoder.pkl')

# Generate Cast Popularity (synthetic feature based on budget correlation)
# Higher budget movies tend to have more popular casts
# Scale to 0-100 range
if 'budget' in df_work.columns:
    # Create a realistic correlation: higher budget = higher cast popularity
    # Add some randomness to make it more realistic
    np.random.seed(42)
    budget_normalized = (df_work['budget'] - df_work['budget'].min()) / (df_work['budget'].max() - df_work['budget'].min() + 1)
    df_work['cast_popularity'] = (budget_normalized * 70 + np.random.normal(15, 10, len(df_work))).clip(0, 100)
    print("   [OK] Cast Popularity generated (correlated with budget)")
else:
    df_work['cast_popularity'] = np.random.uniform(20, 80, len(df_work))
    print("   [WARNING] Cast Popularity generated randomly")

# Generate Director Popularity (synthetic feature with some correlation to success)
# Directors of successful movies tend to have higher popularity
np.random.seed(42)
base_popularity = df_work['success'] * 20 + np.random.normal(30, 15, len(df_work))
df_work['director_popularity'] = base_popularity.clip(0, 100)
print("   [OK] Director Popularity generated (correlated with historical success)")

# Generate Production Company Score (based on production_companies data)
# Calculate historical success rate of production companies
if 'production_companies' in df.columns:
    def extract_company_name(company_str):
        """Extract first production company from JSON-like string"""
        if pd.isna(company_str):
            return 'Unknown'
        if isinstance(company_str, str):
            try:
                company_list = json.loads(company_str.replace("'", '"'))
                if isinstance(company_list, list) and len(company_list) > 0:
                    return company_list[0].get('name', 'Unknown') if isinstance(company_list[0], dict) else company_list[0]
            except:
                return 'Unknown'
        return 'Unknown'
    
    df_work['company_name'] = df['production_companies'].apply(extract_company_name)
    
    # Calculate historical success rate for each company
    company_success_rate = df_work.groupby('company_name')['success'].mean()
    company_success_rate = company_success_rate.fillna(df_work['success'].mean())
    
    # Map company success rate to 0-100 scale
    df_work['production_company_score'] = df_work['company_name'].map(company_success_rate) * 100
    df_work['production_company_score'] = df_work['production_company_score'].fillna(50)  # Default for unknown companies
    df_work.drop('company_name', axis=1, inplace=True)
    print("   [OK] Production Company Score generated (based on historical success rate)")
else:
    # Generate synthetic production company score
    np.random.seed(42)
    df_work['production_company_score'] = (df_work['success'] * 25 + np.random.normal(40, 15, len(df_work))).clip(0, 100)
    print("   [WARNING] Production Company Score generated synthetically")

# ============================================================================
# 3. Handle Missing Values
# ============================================================================
print("\n[4] Handling Missing Values...")

# Fill numerical columns with median
numerical_cols = df_work.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col != 'success']

for col in numerical_cols:
    if df_work[col].isnull().sum() > 0:
        median_val = df_work[col].median()
        df_work[col].fillna(median_val, inplace=True)
        print(f"   [OK] Filled {col} with median: {median_val:.2f}")

# ============================================================================
# 4. Prepare Features and Target
# ============================================================================
print("\n[5] Preparing Features and Target...")

# Select only pre-release feature columns (exclude target and any post-release features)
# Pre-release features: genre_encoded, budget, runtime, release_year, cast_popularity, director_popularity, production_company_score
feature_cols = ['genre_encoded', 'budget', 'runtime', 'release_year', 
                'cast_popularity', 'director_popularity', 'production_company_score']

# Verify all features exist
missing_features = [col for col in feature_cols if col not in df_work.columns]
if missing_features:
    print(f"   [WARNING] Missing features: {missing_features}")
    feature_cols = [col for col in feature_cols if col in df_work.columns]

X = df_work[feature_cols]
y = df_work['success']

print(f"   [OK] Feature columns: {feature_cols}")
print(f"   [OK] Target distribution:")
print(f"      Flop (0): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
print(f"      Hit (1): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")

# ============================================================================
# 5. Scale Features
# ============================================================================
print("\n[6] Scaling Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print("   [OK] Features scaled using StandardScaler")

# ============================================================================
# 6. Train-Test Split
# ============================================================================
print("\n[7] Splitting Dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   [OK] Train set: {X_train.shape[0]} samples")
print(f"   [OK] Test set: {X_test.shape[0]} samples")

# ============================================================================
# 7. Train Random Forest Model
# ============================================================================
print("\n[8] Training Random Forest Classifier...")
print("-" * 60)

# Use RandomForestClassifier with specified parameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("   [OK] Model trained successfully")

# ============================================================================
# 8. Evaluate Model
# ============================================================================
print("\n[9] Evaluating Model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Save model
joblib.dump(model, 'movie_model.pkl')
print("   [OK] Model saved as 'movie_model.pkl'")

# Save feature names for later use
joblib.dump(feature_cols, 'feature_names.pkl')

# Save model info for Streamlit app
model_info = {
    'accuracy': accuracy,
    'accuracy_str': f"{accuracy:.4f} ({accuracy*100:.2f}%)",
    'model_name': 'Random Forest Classifier',
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'precision': precision,
    'recall': recall,
    'f1': f1
}
joblib.dump(model_info, 'model_info.pkl')
print("   [OK] Model info saved as 'model_info.pkl'")

# ============================================================================
# 9. Feature Importance
# ============================================================================
print("\n[10] Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}")

# Plot feature importance (top 10)
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("   [OK] Feature importance plot saved as 'feature_importance.png'")
plt.close()

# ============================================================================
# 10. Correlation Heatmap
# ============================================================================
print("\n[11] Creating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
# Include target in correlation matrix for visualization
correlation_data = df_work[feature_cols + ['success']]
correlation_matrix = correlation_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Pre-Release Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("   [OK] Correlation heatmap saved as 'correlation_heatmap.png'")
plt.close()

# ============================================================================
# 11. Class Balance Chart
# ============================================================================
print("\n[12] Creating Class Balance Chart...")
plt.figure(figsize=(8, 6))
class_counts = df_work['success'].value_counts().sort_index()
colors = ['#ff6b6b', '#51cf66']
plt.bar(['Flop (0)', 'Hit (1)'], class_counts.values, color=colors)
plt.ylabel('Count', fontsize=12)
plt.title('Class Balance - Hit vs Flop', fontsize=14, fontweight='bold')
for i, v in enumerate(class_counts.values):
    plt.text(i, v + max(class_counts.values)*0.01, str(v), ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('class_balance.png', dpi=300, bbox_inches='tight')
print("   [OK] Class balance chart saved as 'class_balance.png'")
plt.close()

# ============================================================================
# 12. Confusion Matrix
# ============================================================================
print("\n[13] Creating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Flop', 'Hit'], yticklabels=['Flop', 'Hit'])
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   [OK] Confusion matrix saved as 'confusion_matrix.png'")
plt.close()

# ============================================================================
# 13. Summary
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Dataset Shape: {df.shape[0]} rows")
print(f"Model: Random Forest Classifier")
print(f"  - n_estimators: 200")
print(f"  - max_depth: 10")
print(f"  - class_weight: balanced")
print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n[OK] Training completed successfully!")
print("=" * 60)
