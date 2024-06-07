# NLP

This repository contains code, data, and plots related to sentiment analysis of financial news headlines from the perspective of a retail investor.

## Folders

1. **codes**: Contains scripts for sentiment analysis.
2. **plots**: Includes visualizations generated during analysis.
3. **data**: Contains the dataset used for sentiment analysis.

## Scripts

### 1. Start_Sentiment_Analysis.py

This script performs sentiment analysis using various machine learning models and evaluates their performance.

- **Description**: This script reads the dataset, preprocesses the text, trains multiple classifiers, evaluates their performance, and generates various plots such as class distribution, model accuracy, precision, recall, and F1 score.
- **Models**: Decision Tree, Logistic Regression, Multinomial Naive Bayes, Linear SVM, Gradient Boosting, Neural Network.

### 2. Augmentation_GridSearch.py

This script augments the text data and performs grid search for hyperparameter tuning.

- **Description**: It augments the text data, performs preprocessing, trains models with augmented data, evaluates their performance, and conducts grid search for parameter optimization.
- **Models**: Decision Tree, Logistic Regression, Multinomial Naive Bayes, Linear SVM, Gradient Boosting, Neural Network.

### 3. Ensemble.py

This script implements an ensemble model using a voting classifier.

- **Description**: It combines multiple models using a voting classifier, performs preprocessing, trains the ensemble model, evaluates its performance, and conducts grid search for parameter optimization.
- **Models**: Logistic Regression, Linear SVM, Multinomial Naive Bayes.

## About Dataset

The dataset (FinancialPhraseBank) contains sentiments for financial news headlines from the perspective of a retail investor. It consists of two columns: "Sentiment" (negative, neutral, or positive) and "News Headline".

## Plots

1. **countplot.png**: Class distribution plot.
2. **pie_chart.png**: Pie chart showing sentiment distribution.
3. **Accuracy.png**: Model accuracy plot.
4. **Precision.png**: Model precision plot.
5. **Recall.png**: Model recall plot.
6. **F1.png**: F1 score plot.
7. **Augmentation_accuracy.png**: Accuracy plot after data augmentation.
8. **GridSearch_accuracy.png**: Accuracy plot after grid search.
9. **ensemble_accuracy.png**: Accuracy plot for the ensemble model.

## Acknowledgements

The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) and was curated by Ankur Sinha. I extend my gratitude to Kaggle and Ankur Sinha for providing this valuable dataset for sentiment analysis of financial news headlines.

## License

This project is licensed under the [MIT License](LICENSE).
