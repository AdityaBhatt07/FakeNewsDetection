#  Fake News Detection using NLP & Machine Learning

##  Project Overview
Fake news has become a serious challenge in the digital era, influencing public opinion and spreading misinformation rapidly.  
This project focuses on **detecting fake news articles** using **Natural Language Processing (NLP)** techniques and **Machine Learning models**.  

The system analyzes news content, source credibility, keyword patterns, sentiment, and textual features to classify news as **Real** or **Fake** with high accuracy.

---

##  Objectives
- Clean and preprocess raw news article data
- Analyze distribution of real vs fake news
- Identify unreliable news sources
- Detect common keywords used in fake news
- Perform sentiment analysis on news articles
- Build and evaluate multiple ML classification models
- Compare model performance using ROC-AUC

---

##  Dataset Description
The dataset contains news articles with the following attributes:

| Column Name | Description |
|------------|------------|
| `title` | Headline of the news article |
| `text` | Full content of the article |
| `site_url` | Source website of the news |
| `label` | Classification (`Real` or `Fake`) |

---

##  Technologies & Libraries Used

### Programming Language
- Python 

### Data Handling & Visualization
- pandas
- matplotlib
- seaborn

### Natural Language Processing
- NLTK
- WordNet Lemmatizer
- VADER Sentiment Analyzer
- WordCloud

### Machine Learning
- scikit-learn
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- XGBoost

---

##  Project Workflow

### 1️⃣ Import Libraries
All essential libraries for data processing, NLP, visualization, and machine learning are imported and configured.

---

### 2️⃣ Dataset Upload
The dataset is uploaded and loaded using **Google Colab**, allowing flexible execution in a cloud environment.

---

### 3️⃣ Data Cleaning
- Removed missing values
- Removed duplicate records
- Ensured clean and consistent data for analysis

---

### 4️⃣ Exploratory Data Analysis (EDA)
- Visualized the distribution of **Real vs Fake news**
- Checked class balance using count plots

---

### 5️⃣ Source Credibility Analysis
- Grouped news by `site_url` and `label`
- Calculated percentage of fake and real news per source
- Identified **Top 10 least credible news sources**

---

### 6️⃣ Keyword Detection in Fake News
- Tokenized text and titles
- Removed stopwords
- Extracted frequently occurring words in fake news
- Visualized:
  - Top 20 fake news keywords (Bar Chart)
  - Fake news Word Cloud

---

### 7️⃣ Text Preprocessing
Performed advanced text cleaning:
- Lowercasing
- Punctuation removal
- Digit removal
- Stopword removal
- Lemmatization

Generated a new feature:
- `clean_text`

---

### 8️⃣ Sentiment Analysis
Used **VADER Sentiment Analyzer** to classify news articles as:
- Positive
- Negative
- Neutral

Visualized sentiment distribution across all articles.

---

### 9️⃣ Feature Extraction
- Applied **TF-IDF Vectorization**
- Limited features to top 5000 terms
- Converted text into numerical form for ML models

---

### 🔟 Machine Learning Models Implemented

The following models were trained and evaluated:

| Model | Description |
|------|------------|
| Logistic Regression | Baseline linear classifier |
| Multinomial Naive Bayes | Probabilistic text classifier |
| Random Forest | Ensemble tree-based model |
| XGBoost | Gradient boosting classifier |

---

###  Model Evaluation Metrics
Each model was evaluated using:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score
- ROC Curve
- AUC Score

Visualization includes:
- Confusion Matrix Heatmaps
- ROC Curves for performance comparison

---

##  Results
- Ensemble models (Random Forest & XGBoost) achieved higher accuracy
- TF-IDF significantly improved text classification performance
- Fake news articles often showed emotionally charged language
- Certain sources consistently produced higher fake content

---

##  Key Learnings
- NLP preprocessing plays a crucial role in model accuracy
- Fake news uses repetitive keywords and emotional tone
- Source-based analysis adds strong credibility insights
- ROC-AUC is more reliable than accuracy for model comparison

---

##  Role & Contribution
As part of this project, the following responsibilities were handled:
- Data cleaning and preprocessing
- Exploratory Data Analysis
- Keyword and sentiment analysis
- Feature engineering using TF-IDF
- Training and evaluation of ML models
- Visualization and result interpretation

---

##  How to Run the Project

1. Open Google Colab or Jupyter Notebook
2. Upload the dataset (`news_articles.csv`)
3. Install required libraries if not available
4. Run the notebook step-by-step
5. View visualizations and model performance outputs

---

##  Future Improvements
- Use deep learning models (LSTM, BERT)
- Add real-time news scraping
- Deploy as a web application
- Improve explainability using SHAP or LIME

---

##  Acknowledgements
- NLTK Team
- Scikit-learn
- Open-source ML community

---
