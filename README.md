## Sentiment Analysis


## Overview
This sentiment analysis project focuses on analyzing Amazon food reviews to extract sentiment information from textual data. The project employs unsupervised methods such as VADER and supervised machine learning models to predict the sentiment of reviews. First, the sentiment of each review is predicted using unsupervised methods, including VADER and DistilBERT, providing scores for positive, neutral, and negative sentiments. Then, the dataset's Score column is transformed into a binary column of positive or negative sentiment, and various classification machine learning models are trained to predict sentiment.

## Features
Unsupervised Sentiment Analysis:
Utilize VADER and DistilBERT to predict the sentiment of reviews.
Outputs scores for positive, neutral, and negative sentiments for each review.
Supervised Sentiment Prediction:
Transform the Score column into a binary column of positive or negative sentiment.
Train classification machine learning models to predict sentiment based on textual data.
Model Evaluation:
Evaluate the performance of each model using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.


## Technologies Used
- Poetry: Dependency management for Python projects.
- Python: Programming language for data preprocessing, model training, and evaluation.
- NLTK: Natural Language Toolkit for text preprocessing and sentiment analysis.
- Scikit-learn: Machine learning library for training and evaluating supervised models.
- Pandas, NumPy: Libraries for data manipulation and analysis.
- Matplotlib, Seaborn: Libraries for data visualization.
- Docker: Containerization for easy deployment and scalability.