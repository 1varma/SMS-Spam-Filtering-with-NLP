Sure, here's a README.md file for your GitHub repository:

---

![Process-of-SMS-Spam-Detection](https://github.com/1varma/SMS-Spam-Filtering-with-NLP/assets/39651154/328e0455-405b-43ae-8f49-abed7525f07c)


## SMS Spam Detection using Machine Learning

This repository contains code for building a machine learning model to detect spam SMS messages using natural language processing (NLP) techniques.

### Requirements

To run the code in this repository, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook (optional)
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`

You can install the required libraries using pip:

```
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```

### Data

The dataset used in this project is the "SMS Spam Collection" dataset, which contains SMS messages labeled as spam or ham (not spam). The dataset is located at `/content/drive/MyDrive/datasets/SMSSpamCollection`.

### Exploratory Data Analysis (EDA)

The EDA process involves understanding the structure and characteristics of the dataset. Key insights from the EDA include:

- Summary statistics of the dataset.
- Distribution of message lengths.
- Visualization of message lengths by label (spam or ham).

### Feature Engineering

Feature engineering involves creating new features from the existing data to improve model performance. In this project, the following feature was engineered:

- Message length: Length of each SMS message.

### Data Visualization

Data visualization is essential for understanding the distribution and relationships within the data. Visualizations included:

- Histogram of message lengths.
- Boxplot of message lengths by label (spam or ham).

### Text Preprocessing

Text preprocessing is crucial for NLP tasks. Steps involved in text preprocessing:

- Removal of punctuation and special characters.
- Removal of stopwords (common words with little semantic value).
- Tokenization: Splitting text into individual words.

### Vectorization

Vectorization converts text data into numerical form suitable for machine learning algorithms. Techniques used:

- Bag of Words (BoW): Representing text data as a matrix of word occurrences.
- Term Frequency-Inverse Document Frequency (TF-IDF): Assigning weights to words based on their frequency in a document and across the entire dataset.

### Model Training

The machine learning model used for spam detection is a Multinomial Naive Bayes classifier. The model is trained using TF-IDF transformed data.

### Model Evaluation

The model's performance is evaluated using metrics such as precision, recall, and F1-score. Additionally, a confusion matrix is generated to visualize the model's predictions.

### Train-Test Split

The dataset is split into training and testing sets to assess the model's generalization performance.

### Creating a Data Pipeline

A data pipeline is created to streamline the process of vectorization, model training, and prediction.

### Conclusion

The final model achieves high accuracy in classifying spam and ham messages. This repository serves as a comprehensive guide for building a spam detection system using machine learning and NLP techniques.
