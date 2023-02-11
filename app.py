#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel
import pickle


app = Flask(__name__)
model=pickle.load(open('sentiment-classification-xg_hpt-boost-model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # getting user from the html form
    user = request.form['userName']

    user = user.lower()
    items = SentimentRecommenderModel().getSentimentRecommendations(user)

    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)

        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message="User Name doesn't exists, No product recommendations at this point of time!")


@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    # getting the review text from the html form
    review_text = request.form["reviewText"]
    print(review_text)
    pred_sentiment = SentimentRecommenderModel().classify_sentiment(review_text)
    print(pred_sentiment)
    return render_template("index.html", sentiment=pred_sentiment)


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




