import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("health_chat_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50  # Should match training

label_map = {
    0: "Malaria is a disease caused by parasites transmitted by mosquitoes.",
    1: "Diabetes is a condition where blood sugar levels are too high.",
    2: "Your symptoms may require a physical exam or lab tests.",
    # Add more based on your model's output classes
}

def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

st.set_page_config(page_title="Health LLM Chatbot", layout="wide")
st.title("🩺 Health LLM Chatbot")
st.write("Ask any health-related question and receive a smart response.")

user_input = st.text_input("You:", "")

if user_input:
    try:
        processed_input = preprocess(user_input)
        prediction = model.predict(processed_input)

        # Debug output
        st.write("Raw prediction:", prediction)

        if prediction.shape[-1] == 1:
            class_index = int(round(prediction[0][0]))
        else:
            class_index = int(np.argmax(prediction[0]))

        response = label_map.get(class_index, "Sorry, I don't understand the question.")
        st.success(f"🤖 Chatbot: {response}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
