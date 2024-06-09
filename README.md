# Fake News Detection using Hugging Face Transformers

This project aims to detect fake news using a fine-tuned BERT-based model. The model is trained on the `GonzaloA/fake_news` dataset from Hugging Face. This project includes data extraction, preprocessing, model training, evaluation, and application to web-scraped news articles.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction on New Articles](#prediction-on-new-articles)
- [Results](#results)

## Installation

Clone the repository and install the necessary packages:
```bash
git clone https://github.com/spoon0525/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

## Usage

### Data Extraction and Preprocessing
Extract text from a news article webpage:
```python
python main.py --url "https://tw.news.yahoo.com/sample-news-article"
```

### Model Training
Train the model on the `GonzaloA/fake_news` dataset:
```python
python main.py --train TRAIN
```

### Evaluation
Evaluate the model performance:
```python
python main.py --train TRAIN --eval EVAL
```

### Prediction on New Articles
Predict whether a new article is fake or real:
```python
python main.py --url "https://tw.news.yahoo.com/sample-news-article"
```

## Dataset
The dataset used in this project is `GonzaloA/fake_news` from Hugging Face. It consists of labeled news articles for training a fake news detection model.

## Model Training
The model is a fine-tuned `distilbert-base-uncased` from Hugging Face. The training script preprocesses the data, tokenizes it, and fine-tunes the model.

## Evaluation
The model's performance is evaluated on a separate test set. The evaluation metrics include accuracy, precision, recall, and F1-score.

## Prediction on New Articles
The `main.py` script fetches an article from a URL, processes the text, and predicts if the article is fake news. It outputs the prediction along with the confidence score.

## Results
Here are some sample results:

![image](https://github.com/spoon0525/fake-news-detection/assets/129286955/87f3a90d-4986-4c68-a2b9-e703b4bf577d)
