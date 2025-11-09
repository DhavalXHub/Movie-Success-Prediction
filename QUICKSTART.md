# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Go to [Kaggle TMDB Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Download `tmdb_5000_movies.csv`
3. Place it in the `dataset/` folder
4. Rename it to `tmdb_movies.csv`

Or check if dataset exists:
```bash
python download_dataset.py
```

### Step 3: Train the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression and Random Forest models
- Compare models and select the best one
- Generate visualizations
- Save the trained model

### Step 4: Run the Streamlit App

**Windows:**
```bash
python -m streamlit run app.py
```
Or double-click `run_app.bat`

**Linux/Mac:**
```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Note:** If `streamlit` command is not recognized, use `python -m streamlit` instead.

### Step 5: Make Predictions
1. Enter movie details in the form
2. Click "Predict Movie Success"
3. View the prediction results

## 📊 Expected Output

After training, you should see:
- Model accuracy and metrics
- Feature importance plot
- Correlation heatmap
- Class balance chart
- Confusion matrix
- Genre success rate (if available)

## 🎯 Sample Prediction

Try these values:
- **Genre**: Action
- **Budget**: 100,000,000
- **Runtime**: 120
- **Popularity**: 15.0
- **Vote Average**: 7.5
- **Vote Count**: 5000
- **Release Year**: 2020

## ❓ Troubleshooting

1. **Dataset not found**: Make sure `tmdb_movies.csv` is in the `dataset/` folder
2. **Model not found**: Run `train_model.py` first
3. **Import errors**: Install all dependencies with `pip install -r requirements.txt`

## 📝 Notes

- The model needs to be trained before using the Streamlit app
- Training typically takes 1-2 minutes depending on dataset size
- All visualizations are saved as PNG files in the project root

---

Happy Predicting! 🎬

