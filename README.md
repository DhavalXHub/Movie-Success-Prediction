# 🎬 Pre-Release Movie Success Prediction

A complete end-to-end machine learning project that predicts whether an upcoming movie will be a **Hit** or **Flop** *before release* using only pre-release data such as budget, genre, runtime, and production attributes.

## 📋 Project Overview

This project uses a real TMDB dataset to train a machine learning model that can predict movie success **before release**. The model excludes post-release features like ratings or votes to avoid data leakage, making it suitable for predicting the success of upcoming movies.

### Key Features
- **Pre-Release Prediction**: Uses only data available before a movie's release
- **No Data Leakage**: Excludes post-release features like ratings, votes, and popularity scores
- **Comprehensive Analysis**: Includes feature importance, visualizations, and model evaluation
- **Interactive Web App**: Streamlit-based interface for easy predictions
- **Production-Ready**: Well-commented code suitable for educational purposes

### Target Variable
- **Hit (1)**: Revenue > Budget × 2
- **Flop (0)**: Otherwise

*Note: Revenue is only used for creating training labels, not as an input feature for predictions.*

## 📊 Dataset Description

The project uses the TMDB (The Movie Database) dataset which contains information about movies including:

### Pre-Release Features (Used for Prediction)
- **budget**: Production budget of the movie
- **runtime**: Duration of the movie in minutes
- **genres**: Genre(s) of the movie
- **release_year**: Year of release
- **cast_popularity**: Estimated popularity of the cast (0-100)
- **director_popularity**: Estimated popularity of the director (0-100)
- **production_company_score**: Historical success rate of the production company (0-100)

### Post-Release Features (Excluded from Model)
- ❌ **popularity**: Popularity score (post-release)
- ❌ **vote_average**: Average rating (post-release)
- ❌ **vote_count**: Number of votes (post-release)
- ❌ **revenue**: Total revenue (only used for training label creation)

## 🛠️ Steps of Data Preprocessing

1. **Data Loading**: Load the TMDB dataset from CSV file
2. **Feature Engineering**:
   - Extract release year from release_date
   - Extract and encode genre information
   - Generate Cast Popularity (correlated with budget)
   - Generate Director Popularity (correlated with historical success)
   - Calculate Production Company Score (based on historical success rate)
   - Create target variable based on revenue and budget (for training only)
3. **Missing Value Handling**:
   - Numerical columns: Fill with median
   - Categorical columns: Fill with most frequent value
4. **Genre Encoding**: Use LabelEncoder to encode genre information
5. **Feature Scaling**: Apply StandardScaler to normalize numerical features
6. **Train-Test Split**: 80/20 split with stratification to maintain class balance

## 🤖 Model Used & Accuracy Achieved

### Random Forest Classifier
The model uses a Random Forest Classifier with the following parameters:
- **n_estimators**: 200
- **max_depth**: 10
- **class_weight**: 'balanced' (to handle class imbalance)
- **random_state**: 42

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The model is saved as `movie_model.pkl` and can be used for predictions on new data.

## 📈 Visualizations Included

1. **Feature Importance Chart**: Top 10 most important features (Random Forest)
2. **Correlation Heatmap**: Shows relationships between all pre-release features
3. **Class Balance Chart**: Distribution of Hit vs Flop in the dataset
4. **Confusion Matrix**: Model performance visualization

All visualizations are saved as PNG files in the project root directory.

## 🚀 How to Run

### Prerequisites

1. **Download the Dataset**:
   - Download the TMDB dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
   - Place `tmdb_5000_movies.csv` in the `dataset/` folder
   - Rename it to `tmdb_movies.csv`
   - Alternatively, run `python download_dataset.py` to check if the dataset exists and get instructions

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Train the Model

Run the training script to preprocess data, train the model, and generate visualizations:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Generate pre-release features
- Train the Random Forest model
- Generate all visualizations
- Save the trained model as `movie_model.pkl`
- Save preprocessors (scaler, encoder) for later use

### Step 2: Run the Streamlit App

Launch the interactive web application:

**Windows:**
```bash
python -m streamlit run app.py
```
Or use the batch file:
```bash
run_app.bat
```

**Linux/Mac:**
```bash
python -m streamlit run app.py
```
Or use the shell script:
```bash
chmod +x run_app.sh
./run_app.sh
```

The app will open in your default web browser at `http://localhost:8501`

**Note:** If `streamlit` command is not recognized, always use `python -m streamlit` instead.

## 🎯 Streamlit App Features

The Streamlit application provides an interactive interface for pre-release movie success prediction:

### Input Fields
- **Genre**: Select from available genres (dropdown)
- **Budget**: Enter movie budget in dollars (numeric input)
- **Runtime**: Enter movie duration in minutes (numeric input)
- **Release Year**: Select release year (numeric input)
- **Cast Popularity**: Estimated popularity of the cast (slider 0-100)
- **Director Popularity**: Estimated popularity of the director (slider 0-100)
- **Production Company Score**: Historical success rate of the production company (slider 0-100)

### Output
- **Prediction**: Hit or Flop
- **Hit Probability**: Percentage probability of being a hit
- **Flop Probability**: Percentage probability of being a flop
- **Model Accuracy**: Trained model accuracy displayed below predictions
- **Insights**: Additional analysis and recommendations

### Features
- Dark-themed, minimal dashboard UI
- Real-time predictions
- Model accuracy and timestamp display
- Responsive design
- Clear probability visualization using `st.metric`

## 📁 Folder Structure

```
ML_project2.0/
│
├── dataset/
│   └── tmdb_movies.csv          # Dataset file (download from Kaggle)
│
├── train_model.py               # Main training script
├── movie_model.pkl              # Trained model (generated)
├── scaler.pkl                   # Feature scaler (generated)
├── genre_encoder.pkl            # Genre encoder (generated)
├── feature_names.pkl            # Feature names (generated)
├── model_info.pkl               # Model information (generated)
│
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── correlation_heatmap.png      # Visualization (generated)
├── class_balance.png            # Visualization (generated)
├── feature_importance.png       # Visualization (generated)
└── confusion_matrix.png         # Visualization (generated)
```

## 🔍 Additional Features

- ✅ Pre-release feature engineering
- ✅ Synthetic feature generation (Cast Popularity, Director Popularity, Production Company Score)
- ✅ Confusion matrix visualization after training
- ✅ Model accuracy and timestamp display in Streamlit UI
- ✅ Feature importance analysis
- ✅ Comprehensive error handling
- ✅ Well-commented code for easy understanding
- ✅ Production Company Score based on historical success rate
- ✅ Class-balanced Random Forest model

## 📝 Notes

- **Pre-Release Focus**: This project predicts movie success before release using only pre-release data
- **No Data Leakage**: Post-release features like ratings and votes are excluded from the model
- **Synthetic Features**: Cast Popularity, Director Popularity, and Production Company Score are generated from available data or created synthetically
- **Random Forest**: Uses Random Forest Classifier as the main model for better performance and feature importance analysis
- **Educational Purpose**: All code is well-commented and designed to be easy to understand for 5th-semester BTech students

## 🐛 Troubleshooting

1. **Dataset not found**: Make sure `tmdb_movies.csv` is in the `dataset/` folder
2. **Missing columns**: The script will handle missing columns gracefully and generate synthetic features
3. **Model not found**: Run `train_model.py` first before using the Streamlit app
4. **Installation issues**: Make sure you're using Python 3.8 or higher
5. **Feature mismatch**: If you get feature mismatch errors, retrain the model using `train_model.py`

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Created as a complete ML project for pre-release movie success prediction.

---

## ⚠️ Important Note

**This project predicts a movie's likely success before its release using only pre-release data such as budget, genre, runtime, and production attributes. The model excludes post-release features like ratings or votes to avoid data leakage.**

---

**Happy Predicting! 🎬🎯**
#   M o v i e - S u c c e s s - P r e d i c t i o n  
 