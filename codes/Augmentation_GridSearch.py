# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import random
import plotly.express as px

# Read data into DataFrame
df = pd.read_csv("D:\\Dell\\repos\\NLP\\data\\all-data.csv", 
                header=None, names=['Sentiment', 'Text'], encoding='latin1')

# Display the first 10 rows of the DataFrame and its length
print(df.head(10))
print(len(df))

# Initialize WordNet lemmatizer and English stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    # Lemmatize tokens and remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to text data
df['Text'] = df['Text'].apply(preprocess_text)

# Define function to augment text
def augment_text(text):
    words = text.split()
    if len(words) > 1:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]  # Swap two adjacent words
    return ' '.join(words)

# Augment text data
augmented_texts = [augment_text(text) for text in df['Text']]
augmented_df = pd.DataFrame({'Sentiment': df['Sentiment'], 'Text': augmented_texts})
df = pd.concat([df, augmented_df])

# Display the length of the DataFrame after augmentation
print(len(df))

# Create a pie chart to visualize the distribution of sentiments
fig = px.pie(df, names='Sentiment', title ='Pie chart of different sentiments of tweets')
fig.write_image("D:\\Dell\\repos\\NLP\\plots\\pie_chart.png")
fig.show()

# Encode sentiment labels
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
                                    df['Text'], df['Sentiment'], 
                                    test_size=0.2, random_state=42)

# Define machine learning models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}

# Dictionary to store accuracy before grid search
accuracies_before_grid_search = {}

# Iterate over models, train, and evaluate
for name, model in models.items():
    print(f"Training {name}...")
    # Create a pipeline for text classification
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', model)
    ])
    text_clf.fit(X_train, y_train)
    predictions = text_clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracies_before_grid_search[name] = accuracy

# Plot accuracy before grid search
plt.figure(figsize=(10, 6))
plt.bar(accuracies_before_grid_search.keys(), accuracies_before_grid_search.values(), color='skyblue')
plt.title('Accuracy After Augmentation')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\Augmentation_accuracy.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Define parameter grids for grid search
param_grid_lr = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__solver': ['liblinear', 'saga']
}

param_grid_svm = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__loss': ['hinge', 'squared_hinge']
}

# Grid search for Logistic Regression
print("Training Logistic Regression with Grid Search...")
text_clf_lr = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])
grid_search_lr = GridSearchCV(text_clf_lr, param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
best_params_lr = grid_search_lr.best_params_
predictions_lr = grid_search_lr.predict(X_test)
report_lr = classification_report(y_test, predictions_lr, target_names=label_encoder.classes_)
print(f"Best parameters for Logistic Regression: {best_params_lr}")
print(f"Evaluation results for Logistic Regression:")
print(report_lr)

# Grid search for Linear SVM
print("Training Linear SVM with Grid Search...")
text_clf_svm = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LinearSVC())
])
grid_search_svm = GridSearchCV(text_clf_svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_params_svm = grid_search_svm.best_params_
predictions_svm = grid_search_svm.predict(X_test)
report_svm = classification_report(y_test, predictions_svm, target_names=label_encoder.classes_)
print(f"Best parameters for Linear SVM: {best_params_svm}")
print(f"Evaluation results for Linear SVM:")
print(report_svm)

# Dictionary to store accuracy after grid search
accuracies_after_grid_search = {
    'Logistic Regression': accuracy_score(y_test, predictions_lr),
    'Linear SVM': accuracy_score(y_test, predictions_svm)
}

# Plot accuracy after grid search
plt.figure(figsize=(10, 6))
plt.bar(accuracies_after_grid_search.keys(), accuracies_after_grid_search.values(), color='lightgreen')
plt.title('Accuracy After Grid Search')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\GridSearch_accuracy.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()
