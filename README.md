# 🎬 Pre-Release Movie Success Prediction

🚀 **Live Demo:** https://movie-success-prediction-gmdpg2ts8mtl8fzriqkujh.streamlit.app/

A complete end-to-end machine learning project that predicts whether an upcoming movie will be a **Hit** or **Flop** *before release* using only pre-release data such as budget, genre, runtime, and production attributes.

## 📋 Project Overview

This project uses a real TMDB dataset to train a machine learning model that can predict movie success **before release**. The model excludes post-release features like ratings or votes to avoid data leakage, making it suitable for predicting the success of upcoming movies.

### Key Features

* **Pre-Release Prediction**: Uses only data available before a movie's release
* **No Data Leakage**: Excludes post-release features like ratings, votes, and popularity scores
* **Comprehensive Analysis**: Includes feature importance, visualizations, and model evaluation
* **Interactive Web App**: Streamlit-based interface for easy predictions
* 🌐 **Live Deployed Application** available online
* **Production-Ready**: Well-commented code suitable for educational purposes

### Target Variable

* **Hit (1)**: Revenue > Budget × 2
* **Flop (0)**: Otherwise

*Note: Revenue is only used for creating training labels, not as an input feature for predictions.*

## 📊 Dataset Description

The project uses the TMDB (The Movie Database) dataset which contains information about movies including:

### Pre-Release Features (Used for Prediction)

* **budget**: Production budget of the movie
* **runtime**: Duration of the movie in minutes
* **genres**: Genre(s) of the movie
* **release_year**: Year of release
* **cast_popularity**: Estimated popularity of the cast (0-100)
* **director_popularity**: Estimated popularity of the director (0-100)
* **production_company_score**: Historical success rate of the production company (0-100)

### Post-Release Features (Excluded from Model)

* ❌ **popularity**
* ❌ **vote_average**
* ❌ **vote_count**
* ❌ **revenue**

## 🛠️ Steps of Data Preprocessing

1. Data loading from CSV
2. Feature engineering (year, genres, synthetic scores)
3. Missing value handling
4. Encoding categorical variables
5. Feature scaling
6. Train-test split (80/20)

## 🤖 Model Used & Accuracy Achieved

### Random Forest Classifier

* n_estimators: 200
* max_depth: 10
* class_weight: balanced

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Model saved as `movie_model.pkl`

## 📈 Visualizations Included

* Feature Importance
* Correlation Heatmap
* Class Balance
* Confusion Matrix

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model

```bash
python train_model.py
```

### Run app

```bash
python -m streamlit run app.py
```

## 🎯 Live Application

👉 Try it here:
https://movie-success-prediction-gmdpg2ts8mtl8fzriqkujh.streamlit.app/

## 📁 Folder Structure

```
ML_project2.0/
├── dataset/
├── train_model.py
├── movie_model.pkl
├── app.py
├── requirements.txt
├── README.md
```

## 🔍 Additional Features

* Pre-release feature engineering
* Synthetic feature generation
* Feature importance analysis
* Clean UI with Streamlit
* Error handling

## 📝 Notes

* Uses only pre-release data
* Avoids data leakage
* Designed for learning ML concepts

## 🐛 Troubleshooting

* Ensure dataset is placed correctly
* Run training before app
* Use Python 3.8+

## 📄 License

Educational use only

---

## ⚠️ Important Note

This project predicts movie success using only pre-release data and excludes post-release features.

---

**Happy Predicting! 🎬🎯**
