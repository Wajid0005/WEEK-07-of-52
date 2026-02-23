# 📩 SMS Spam Classifier: ML Pipeline & Web App

**Live Demo:** 👉 [https://sms-spam-classifier-wajid-iqbal-005.streamlit.app/](https://sms-spam-classifier-wajid-iqbal-005.streamlit.app/)

A sleek, lightweight web application that classifies SMS messages as **Spam** or **Not Spam**. This project goes beyond just a basic deployment—it features a robust Natural Language Processing (NLP) pipeline and a rigorously evaluated Machine Learning backend.

---

## 🧠 The Machine Learning Pipeline (Under the Hood)
This application doesn't just guess; it relies on a strict, sequential pipeline to process human text into machine-readable numbers, and then classifies it based on learned probabilities. If you want to understand the backend, here is the 3-step architecture:

### 1. Data Preprocessing (The `text_transform` Engine)
Raw SMS data is notoriously messy. Before any model can learn from it, the text must be standardized. Our custom `text_transform` function applies a 5-step NLP cleaning process:
* **Lowercasing:** Converts all text to lowercase to prevent the model from treating "FREE" and "free" as different words.
* **Tokenization:** Uses NLTK (`word_tokenize`) to break the message down into individual words (tokens).
* **Alphanumeric Filtering:** Strips away isolated special characters and retains only letters and numbers.
* **Stopword & Punctuation Removal:** Drops common English filler words (e.g., "is", "the", "and") and punctuation that carry no predictive weight for spam detection.
* **Stemming:** Applies the NLTK `PorterStemmer` to reduce words to their root form (e.g., "winning", "winner", and "won" all become "win"). This drastically reduces the dimensionality of our dataset.

### 2. Feature Extraction (TF-IDF Vectorization)
Algorithms cannot understand text; they need numbers. Once the text is cleaned, it is passed through a **Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer**. 
* Unlike a simple Bag-of-Words that just counts word frequency, TF-IDF penalizes words that appear too frequently across *all* messages and rewards words that are highly specific to the current message.
* The fitted vectorizer is serialized into `vectorizer.pkl` so the web app can transform new, unseen user input in real-time exactly as it did during training.

### 3. Model Evaluation & Selection (The "Best of 3")
During the backend training phase, the prepared dataset was tested across three distinct classification models to find the optimal balance of precision and accuracy (precision is critical in spam detection to minimize false positives—we don't want important messages going to the spam folder!).
* **Model 1 (e.g., Gaussian Naive Bayes)** * **Model 2 (e.g., Multinomial Naive Bayes)** * **Model 3 (e.g., Bernoulli Naive Bayes / Random Forest)** **Multinomial Naive Bayes** was ultimately selected as the champion model due to its superior performance with discrete text data and TF-IDF features. The winning model was serialized into `model.pkl` for immediate inference in the Streamlit app.

---

## 💻 Core Logic & Code Highlights

Below is the exact backend logic powering the application, highlighting the NLP text transformation and the Streamlit prediction flow.

```python
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize NLTK components
ps = PorterStemmer()

# --- 1. THE NLP PREPROCESSING PIPELINE ---
def text_transform(text):
    text = text.lower() # Lowercase
    text = nltk.word_tokenize(text) # Tokenize

    y = []
    for i in text:
        if i.isalnum(): # Remove special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # Remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        # Apply Porter Stemming
        y.append(ps.stem(i))

    return " ".join(y)

# --- 2. LOAD SERIALIZED BACKEND ---
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- 3. WEB APP UI & PREDICTION LOGIC ---
st.title("SMS Spam Classifier")

input_sms = st.text_area("Paste your message here:")

if st.button('Check'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Step 1: Clean the user input
        transformed_sms = text_transform(input_sms)
        
        # Step 2: Vectorize the cleaned text
        vector_input = tfidf.transform([transformed_sms])
        
        # Step 3: Predict using the best-performing model
        result = model.predict(vector_input)[0]

        # Step 4: Display Output
        if result == 1:
            st.error("Result: Spam")
        else:
            st.success("Result: Not Spam")
