from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def judge(model_path, url):
    # load pre-train model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    import requests
    from bs4 import BeautifulSoup

    def get_news_text(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the main content of the article. This will vary depending on the website structure.
        # You might need to inspect the HTML of the webpage to find the correct tags and classes.
        #article = soup.find('div', {'class': 'tagFormating'})
        paragraphs = soup.find_all('p')

        # Combine the paragraphs into a single string
        news_text = ' '.join([para.get_text() for para in paragraphs])
        return news_text

    news_text = get_news_text(url)

    # input text
    def preprocess(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        return inputs

    input_text = news_text
    inputs = preprocess(input_text)

    # predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate the probability of each category
    probs = F.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=-1)

    # show the result
    labels = ["Real", "Fake"]
    predicted_label = labels[predicted_class.item()]
    confidence_score = confidence.item()

    print(f"result: {predicted_label}")
    print(f"confidence: {confidence_score:.4f}")