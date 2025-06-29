import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("health_chat_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50  # Adjust based on training

# Preprocess input text
def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# Decode output tokens to readable text
def decode_output(predicted_sequence):
    predicted_sequence = np.argmax(predicted_sequence, axis=-1)  # shape: (1, max_len)
    return tokenizer.sequences_to_texts(predicted_sequence)[0]

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
