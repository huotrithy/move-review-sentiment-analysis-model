from enum import Enum
import streamlit as st
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

MODEL_PATH = st.secrets.model.model_path
TOKENIZER_PATH = st.secrets.model.tokenizer_path


LABEL_MAP = {0: "negative", 1: "positive"}


class SentimentLabel(Enum):
    NEGATIVE = "negative"
    POSITIVE = "positive"


st.title("Movie sentiment analysis")
st.write("This is playground for sentiment analysis.")
st.markdown(
    """
- Write your review in the text box below
- Click predict
- AI is will output your review as Negative or Positive
"""
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = ort.InferenceSession(MODEL_PATH)

    return model, tokenizer


model, tokenizer = load_model()


def predict(input):
    if not input:
        return

    encoded_input = tokenizer(
        input,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="np",
    )

    # ONNX works with NumPy arrays, so convert PyTorch tensors to NumPy
    ort_input = {
        "input_ids": encoded_input["input_ids"],
        "attention_mask": encoded_input["attention_mask"],
        "token_type_ids": encoded_input["token_type_ids"],
    }

    ort_outputs = model.run(None, ort_input)

    logits = ort_outputs[0]
    probabilities = torch.softmax(torch.tensor(logits), dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()

    return probabilities, LABEL_MAP[predicted_index]


input_review = st.text_area("Movie review: ", placeholder="This move is good")
if st.button("Predict"):
    probabilities, prediction = predict(input_review)
    st.write(probabilities)
    if prediction == SentimentLabel.POSITIVE.value:
        st.success(prediction)
    else:
        st.error(prediction)
