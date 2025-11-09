"""
Project Verification Script
Checks if all required files and components are present
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} (NOT FOUND)")
        return False

def main():
    print("=" * 60)
    print("🔍 Project Verification")
    print("=" * 60)
    print()
    
    files_to_check = [
        ("train_model.py", "Training script"),
        ("app.py", "Streamlit application"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "README documentation"),
        ("QUICKSTART.md", "Quick start guide"),
        ("download_dataset.py", "Dataset download helper"),
        (".gitignore", "Git ignore file"),
    ]
    
    dirs_to_check = [
        ("dataset", "Dataset directory"),
    ]
    
    print("Checking Files:")
    print("-" * 60)
    file_results = []
    for filepath, description in files_to_check:
        file_results.append(check_file(filepath, description))
    
    print("\nChecking Directories:")
    print("-" * 60)
    dir_results = []
    for dirpath, description in dirs_to_check:
        dir_results.append(check_directory(dirpath, description))
    
    print("\nChecking Dataset:")
    print("-" * 60)
    dataset_exists = check_file("dataset/tmdb_movies.csv", "TMDB dataset")
    
    print("\nChecking Generated Files (after training):")
    print("-" * 60)
    generated_files = [
        ("movie_model.pkl", "Trained model"),
        ("scaler.pkl", "Feature scaler"),
        ("genre_encoder.pkl", "Genre encoder"),
        ("feature_names.pkl", "Feature names"),
        ("model_info.pkl", "Model information"),
    ]
    
    generated_results = []
    for filepath, description in generated_files:
        result = check_file(filepath, description)
        generated_results.append(result)
        if not result:
            print(f"   ℹ️  This file will be created when you run train_model.py")
    
    print("\n" + "=" * 60)
    all_files_ok = all(file_results) and all(dir_results)
    all_generated_ok = all(generated_results)
    
    if all_files_ok:
        print("✅ All required project files are present!")
        if not dataset_exists:
            print("\n⚠️  Dataset not found. Please download it from Kaggle.")
            print("   Run: python download_dataset.py for instructions")
        if not all_generated_ok:
            print("\nℹ️  Some generated files are missing.")
            print("   Run: python train_model.py to generate them")
        print("\n🚀 Project is ready to use!")
    else:
        print("❌ Some required files are missing!")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()

