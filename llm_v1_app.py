import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained Keras model
model = tf.keras.models.load_model("health_chat_model.keras")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define max length (should match what was used during training)
MAX_LEN = 50  # <-- Change this if different during training

# Preprocess user input
def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

# Streamlit app UI
st.set_page_config(page_title="Health LLM Chatbot", layout="wide")
st.title("🩺 Health LLM Chatbot")
st.write("Ask any health-related question and receive a smart response from your custom-trained model.")

# User input
user_input = st.text_input("You:", "")

# Handle input and generate response
if user_input:
    try:
        processed_input = preprocess(user_input)
        prediction = model.predict(processed_input)

        # Convert prediction to readable format (customize based on model output type)
        if prediction.shape[-1] == 1:
            response = prediction[0][0]
        else:
            response = np.argmax(prediction[0])

        st.success(f"🤖 Chatbot: {response}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")