from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html', color="#f0f0f0")  # default color

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])

    # Predict sentiment and confidence
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = round(max(probability) * 100, 2)

    # Choose color based on sentiment
    if prediction.lower() == 'positive':
        color = "#21d34b"  # light green
    else:
        color = "#e01324"  # light red

    # Render template with results
    return render_template(
        'index.html',
        prediction_text=f"Sentiment: {prediction.capitalize()} ({confidence}% confidence)",
        color=color
    )

if __name__ == '__main__':
    app.run(debug=True)
