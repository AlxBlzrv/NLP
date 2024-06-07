# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Read data into DataFrame
df = pd.read_csv("D:\\Dell\\repos\\NLP\\data\\all-data.csv", 
                header=None, names=['Sentiment', 'Text'], encoding='latin1')

print(df.head(10))

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sentiment')
plt.title('Class Distribution')
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\countplot.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Define a lemmatizer function to preprocess text
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to text data
df['Text'] = df['Text'].apply(preprocess_text)

# Encode sentiment labels
label_encoder = LabelEncoder()
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Define machine learning models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}

# Dictionary to store evaluation metrics
metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

# Iterate over models, train, and evaluate
for name, model in models.items():
    print(f"Training {name}...")
    # Create a pipeline for text classification
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', model)
    ])
    text_clf.fit(X_train, y_train)
    predictions = text_clf.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    # Store metrics in the dictionary
    metrics['Model'].append(name)
    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1 Score'].append(f1)
    
    # Print classification report
    report = classification_report(y_test, predictions, target_names=label_encoder.classes_)
    print(f"Evaluation results for {name}:")
    print(report)

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame(metrics)

# Plot accuracy for each model
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Accuracy', data=metrics_df, palette='viridis')
plt.title('Model Accuracy')
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\Accuracy.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot precision for each model
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Precision', data=metrics_df, palette='viridis')
plt.title('Model Precision')
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\Precision.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot recall for each model
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Recall', data=metrics_df, palette='viridis')
plt.title('Model Recall')
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\Recall.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot F1 score for each model
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='F1 Score', data=metrics_df, palette='viridis')
plt.title('Model F1 Score')
plt.savefig("D:\\Dell\\repos\\NLP\\plots\\F1.png", 
            bbox_inches='tight', pad_inches=0.1)
plt.show()
