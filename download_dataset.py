"""
Helper script to download TMDB dataset
This script provides instructions and can help download the dataset from Kaggle
"""

import os
import sys

def check_dataset():
    """Check if dataset exists"""
    dataset_path = "dataset/tmdb_movies.csv"
    if os.path.exists(dataset_path):
        print("✅ Dataset found at:", dataset_path)
        return True
    else:
        print("❌ Dataset not found at:", dataset_path)
        return False

def print_instructions():
    """Print instructions for downloading the dataset"""
    print("\n" + "="*60)
    print("📥 Dataset Download Instructions")
    print("="*60)
    print("\n1. Download the TMDB dataset from Kaggle:")
    print("   URL: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
    print("\n2. The dataset file should be named: tmdb_5000_movies.csv")
    print("\n3. Place the file in the 'dataset' folder")
    print("\n4. Rename it to: tmdb_movies.csv")
    print("\nAlternatively, you can use the Kaggle API:")
    print("   kaggle datasets download -d tmdb/tmdb-movie-metadata")
    print("   unzip tmdb-movie-metadata.zip")
    print("   mv tmdb_5000_movies.csv dataset/tmdb_movies.csv")
    print("\n" + "="*60)

if __name__ == "__main__":
    if not check_dataset():
        print_instructions()
        sys.exit(1)
    else:
        print("✅ Dataset is ready!")
        sys.exit(0)

