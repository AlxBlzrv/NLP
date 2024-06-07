# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import random

# Read data into DataFrame
df = pd.read_csv("D:\\Dell\\repos\\NLP\\data\\all-data.csv", 
                header=None, names=['Sentiment', 'Text'], encoding='latin1')

# Display the first 10 rows of the DataFrame and its length
print(df.head(10))
print(len(df))

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function for text
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to text data
df['Text'] = df['Text'].apply(preprocess_text)

# Data augmentation function
def augment_text(text):
    words = text.split()
    if len(words) > 1:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]  # Swap two adjacent words
    return ' '.join(words)

# Augment the data
augmented_texts = [augment_text(text) for text in df['Text']]
augmented_df = pd.DataFrame({'Sentiment': df['Sentiment'], 'Text': augmented_texts})
df = pd.concat([df, augmented_df])

# Encode sentiment labels
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Define models for ensemble
models = [
    ('logreg', LogisticRegression(max_iter=1000)),
    ('svc', LinearSVC()),
    ('mnb', MultinomialNB())
]

# Create a VotingClassifier ensemble
ensemble_model = VotingClassifier(estimators=models, voting='hard')

# Define parameter grid for grid search
param_grid = {
    'clf__logreg__C': [0.1, 1, 10],
    'clf__svc__C': [0.1, 1, 10]
}

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', ensemble_model)
])

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
predictions = grid_search.predict(X_test)

# Classification report
report = classification_report(y_test, predictions, target_names=label_encoder.classes_)
print(f"Best parameters for the ensemble model: {best_params}")
print("Evaluation results for the ensemble model:")
print(report)

# Get cross-validation results for models
logreg_scores = grid_search.cv_results_['mean_test_score'][grid_search.cv_results_['param_clf__logreg__C'] == best_params['clf__logreg__C']]
svc_scores = grid_search.cv_results_['mean_test_score'][grid_search.cv_results_['param_clf__svc__C'] == best_params['clf__svc__C']]

# Plot accuracy for each model
C_values = [0.1, 1, 10]
plt.figure(figsize=(10, 6))
plt.plot(C_values, logreg_scores, marker='o', label='Logistic Regression')
plt.plot(C_values, svc_scores, marker='o', label='Linear SVC')
plt.title('Model Accuracy vs C Values')
plt.xlabel('C Value')
plt.ylabel('Mean Accuracy')
plt.xscale('log')
plt.xticks(C_values)
plt.legend()
plt.grid(True)
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\ensemble_accuracy.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()
