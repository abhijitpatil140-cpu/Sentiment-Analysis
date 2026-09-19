# 🎬 Sentiment Analysis on Movie Reviews

A machine learning project that classifies movie reviews into **Positive**, **Neutral**, or **Negative** sentiment using Natural Language Processing (NLP) and a **Logistic Regression** classifier with **TF-IDF** features.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Highlights](#-project-highlights)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Evaluation](#-model-evaluation)
- [Predicting Custom Reviews](#-predicting-custom-reviews)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

This project walks through a complete, end-to-end NLP pipeline for sentiment classification of movie reviews:

1. Load and clean the review data
2. Convert numeric ratings into sentiment labels
3. Preprocess the raw text (lowercasing, regex cleaning, stopword removal, lemmatization)
4. Extract features with TF-IDF (unigrams + bigrams)
5. Train a Logistic Regression model
6. Evaluate with accuracy, precision, recall, F1-score, and a confusion matrix
7. Predict the sentiment of new, custom reviews

## ✨ Project Highlights

| Item | Details |
|------|---------|
| **Task** | Multi-class text classification (3 classes) |
| **Algorithm** | Logistic Regression (`max_iter=1000`) |
| **Vectorizer** | TF-IDF (`max_features=500`, `ngram_range=(1,2)`) |
| **Target Classes** | Positive, Neutral, Negative |
| **Train/Test Split** | 80% training / 20% testing (stratified, `random_state=42`) |
| **Language / Environment** | Python (Jupyter Notebook) |

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Confusion matrix heatmap |
| `nltk` | Stopwords and lemmatization |
| `scikit-learn` | TF-IDF, Logistic Regression, evaluation metrics |
| `re` | Regex-based text cleaning (built-in) |

## 📂 Dataset

- **File:** `Movies_Reviews_modified_version1.csv`
- **Columns used:**
  - `Reviews` – the review text
  - `Ratings` – the numeric rating given by the reviewer
- Rows with missing values are dropped during preprocessing.

### Sentiment Labelling Rule

| Rating | Sentiment |
|--------|-----------|
| ≥ 4 | Positive |
| = 3 | Neutral |
| < 3 | Negative |

> The thresholds can be adjusted to suit your dataset's rating distribution.

## 🔄 Project Workflow

```
Raw CSV  →  Clean & Select Columns  →  Rating → Sentiment Label
   →  Text Preprocessing  →  TF-IDF Vectorization  →  Train/Test Split
   →  Logistic Regression  →  Evaluation  →  Custom Predictions
```

### 1. Data Loading & Cleaning
Only the `Reviews` and `Ratings` columns are kept, and missing values are removed with `dropna()`.

### 2. Sentiment Label Generation
A `get_sentiment()` function maps each numeric rating to `positive`, `neutral`, or `negative`, creating the target column `Sentiment`.

### 3. Text Preprocessing
Each review passes through the following steps:

- Convert to lowercase
- Remove URLs
- Remove numbers
- Remove punctuation
- Tokenize on whitespace
- Remove English stopwords
- Lemmatize with `WordNetLemmatizer`

The result is stored in a new `Clean_Review` column.

### 4. Feature Extraction & Split
- `TfidfVectorizer(max_features=500, ngram_range=(1, 2))` converts text into numerical features, capturing both single words and two-word phrases.
- `train_test_split(test_size=0.2, random_state=42, stratify=y)` gives a reproducible 80/20 split that preserves class proportions.

### 5. Model Training
A `LogisticRegression(max_iter=1000)` model is trained on the TF-IDF features.

### 6. Evaluation
The model is evaluated on the held-out 20% test set (see [Model Evaluation](#-model-evaluation)).

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter
```

**4. Download the required NLTK data**

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage

1. Place `Movies_Reviews_modified_version1.csv` in the project root (or update the file path in the notebook).
2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open the notebook (e.g., `Sentiment_Analysis.ipynb`) and run all cells from top to bottom.

## 📊 Model Evaluation

The model is evaluated on the 20% test set using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall percentage of correctly classified reviews |
| **Precision** | Of the reviews predicted as a class, how many were correct |
| **Recall** | Of the reviews truly in a class, how many were found |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Breakdown of correct/incorrect predictions per class, visualized as a seaborn heatmap |

```python
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Results

> Replace the values below with the output from your notebook.

| Metric | Value |
|--------|-------|
| Accuracy | `XX.XX%` |

<!-- Optional: add a screenshot of your confusion matrix -->
<!-- ![Confusion Matrix](images/confusion_matrix.png) -->

## 🔮 Predicting Custom Reviews

```python
sample = 'The movie was fantastic. Acting was excellent.'
vec = tfidf.transform([preprocess_text(sample)])
print('Predicted:', model.predict(vec)[0])
```

**Output:**

```
Predicted: positive
```

## 📁 Project Structure

```
├── Sentiment_Analysis.ipynb                # Main Jupyter Notebook
├── Movies_Reviews_modified_version1.csv    # Dataset
├── Sentiment_Analysis_ML_Report.pdf        # Detailed project report
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

> Adjust file names to match your repository.

## 🔭 Future Improvements

- Increase `max_features` or tune TF-IDF parameters
- Compare with other models (Naive Bayes, SVM, Random Forest)
- Hyperparameter tuning with `GridSearchCV`
- Handle class imbalance (e.g., class weights or resampling)
- Try deep learning approaches (LSTM, BERT) for better contextual understanding
- Deploy as a web app using Streamlit or Flask

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository, create a feature branch, and open a pull request.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

⭐ If you found this project helpful, consider giving it a star!
