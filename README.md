# Fake News Detection System

## Overview
This project detects whether a news article is real or fake using Machine Learning and Natural Language Processing (NLP).

## Key Features
- Preprocesses text using NLP techniques
- Converts text into numerical features using TF-IDF
- Uses Logistic Regression for classification
- Provides real-time prediction through a Streamlit web app

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## How It Works
1. Input news text
2. Text is cleaned and preprocessed
3. TF-IDF converts text to vectors
4. Model predicts whether news is fake or real

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   streamlit run fake_news_detector.py

## Dataset
- LIAR dataset used for training and testing

## Author
M Surendra Reddy
## Demo
![App Screenshot](screenshot.png)
