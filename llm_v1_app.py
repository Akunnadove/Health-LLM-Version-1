import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("health_chat_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50  # Adjust this based on how you trained the model

# Preprocess input text
def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# Decode output tokens to readable text
def decode_output(predicted_sequence):
    predicted_ids = np.argmax(predicted_sequence, axis=-1)[0]  # shape: (max_len,)
    index_word = {index: word for word, index in tokenizer.word_index.items()}
    words = [index_word.get(idx, '') for idx in predicted_ids if idx != 0]
    return ' '.join(words)

# Streamlit app layout
st.set_page_config(page_title="Health LLM Chatbot", layout="wide")
st.title("🩺 Health LLM Chatbot")
st.write("Ask any health-related question and get a smart medical response.")

user_input = st.text_input("You:", "")

if user_input:
    try:
        input_seq = preprocess(user_input)
        prediction = model.predict(input_seq)

        # Convert predicted sequence to readable response
        response_text = decode_output(prediction)

        st.success(f"🤖 Chatbot: {response_text}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
