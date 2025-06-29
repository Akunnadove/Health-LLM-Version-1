import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("health_chat_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Parameters
sequence_length = 5  # This must match training

# Function to generate response
def generate_text(seed_text, next_words=30):
    result = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, truncating='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=-1)[0]
        next_word = tokenizer.index_word.get(predicted_index, '')
        result += " " + next_word
    return result

# Streamlit UI
st.set_page_config(page_title="Health Chatbot 💬", layout="centered")
st.title("🩺 Health Chatbot")
st.markdown("Ask any health-related question below:")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input field
user_input = st.chat_input("Type your question here...")

# When user submits
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_text(user_input, next_words=30)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
