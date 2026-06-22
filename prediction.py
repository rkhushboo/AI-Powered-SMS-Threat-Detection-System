"""
Prediction module for TextCNN integration.
"""
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "TextCNN": BASE_DIR / "best_spam_model_cnn.keras",
    "BiLSTM": BASE_DIR / "best_BiLSTM.keras",
    "BiGRU": BASE_DIR / "best_BiGRU.keras",
    "ANN": BASE_DIR / "best_spam_model_ann.keras",
}
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
THRESHOLD_PATH = BASE_DIR / "threshold.pkl"
CONFIG_PATH = BASE_DIR / "config.json"


@st.cache_resource(show_spinner=False)
def load_textcnn():
    return keras_load_model(MODEL_PATHS["TextCNN"])


@st.cache_resource(show_spinner=False)
def load_bilstm():
    return keras_load_model(MODEL_PATHS["BiLSTM"])


@st.cache_resource(show_spinner=False)
def load_bigru():
    return keras_load_model(MODEL_PATHS["BiGRU"])


@st.cache_resource(show_spinner=False)
def load_ann():
    return keras_load_model(MODEL_PATHS["ANN"])


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_threshold():
    with open(THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)
    if isinstance(threshold, dict):
        threshold = threshold.get("threshold", threshold)
    return float(threshold)


@st.cache_resource(show_spinner=False)
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_confidence_level(probability: float) -> str:
    if probability >= 0.90:
        return "Very High"
    if probability >= 0.75:
        return "High"
    if probability >= 0.60:
        return "Medium"
    return "Low"


def _extract_probability(prediction: np.ndarray) -> float:
    if prediction.ndim == 2 and prediction.shape[1] == 2:
        return float(prediction[0, 1])
    return float(prediction.flatten()[0])


def _prepare_sequence(text: str, tokenizer, max_length: int):
    cleaned_text = preprocess_text(text)
    token_count = len(cleaned_text.split())
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    return cleaned_text, token_count, padded_sequence


def _predict_model(
    model_name: str,
    model,
    tokenizer,
    threshold: float,
    max_length: int,
    text: str,
) -> Dict[str, Any]:
    cleaned_text, token_count, padded_sequence = _prepare_sequence(text, tokenizer, max_length)
    start_time = time.perf_counter()
    raw_output = model.predict(padded_sequence, verbose=0)
    inference_time = int((time.perf_counter() - start_time) * 1000)
    probability = _extract_probability(np.array(raw_output))
    probability = float(np.clip(probability, 0.0, 1.0))
    prediction = "SPAM" if probability >= threshold else "HAM"
    confidence_level = get_confidence_level(probability)

    return {
        "model_name": model_name,
        "prediction": prediction,
        "probability": probability,
        "confidence_score": probability,
        "confidence_level": confidence_level,
        "inference_time": inference_time,
        "threshold": threshold,
        "cleaned_text": cleaned_text,
        "token_count": token_count,
        "sequence_shape": padded_sequence.shape,
        "raw_probability": float(probability),
    }


def predict_textcnn(text: str) -> Dict[str, Any]:
    tokenizer = load_tokenizer()
    threshold = load_threshold()
    config = load_config()
    model = load_textcnn()
    max_length = int(config.get("max_seq_length", 40))
    return _predict_model("TextCNN", model, tokenizer, threshold, max_length, text)


def predict_bilstm(text: str) -> Dict[str, Any]:
    tokenizer = load_tokenizer()
    threshold = load_threshold()
    config = load_config()
    model = load_bilstm()
    max_length = int(config.get("max_seq_length", 40))
    return _predict_model("BiLSTM", model, tokenizer, threshold, max_length, text)


def predict_bigru(text: str) -> Dict[str, Any]:
    tokenizer = load_tokenizer()
    threshold = load_threshold()
    config = load_config()
    model = load_bigru()
    max_length = int(config.get("max_seq_length", 40))
    return _predict_model("BiGRU", model, tokenizer, threshold, max_length, text)


def predict_ann(text: str) -> Dict[str, Any]:
    tokenizer = load_tokenizer()
    threshold = load_threshold()
    config = load_config()
    model = load_ann()
    max_length = int(config.get("max_seq_length", 40))
    return _predict_model("ANN", model, tokenizer, threshold, max_length, text)


def predict_all_models(text: str) -> Dict[str, Dict[str, Any]]:
    functions = {
        "TextCNN": predict_textcnn,
        "BiLSTM": predict_bilstm,
        "BiGRU": predict_bigru,
        "ANN": predict_ann,
    }
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn, text): name for name, fn in functions.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {
                    "model_name": name,
                    "prediction": "ERROR",
                    "probability": 0.0,
                    "confidence_score": 0.0,
                    "confidence_level": "Error",
                    "inference_time": 0,
                    "threshold": load_threshold(),
                    "cleaned_text": "",
                    "token_count": 0,
                    "sequence_shape": (0, 0),
                    "raw_probability": 0.0,
                    "error": str(exc),
                }
    return results


def compute_consensus(model_results: Dict[str, Dict[str, Any]]) -> Tuple[str, int]:
    spam_votes = sum(1 for result in model_results.values() if result.get("prediction") == "SPAM")
    ham_votes = sum(1 for result in model_results.values() if result.get("prediction") == "HAM")
    if spam_votes > ham_votes:
        consensus = "SPAM"
    elif ham_votes > spam_votes:
        consensus = "HAM"
    else:
        consensus = "TIED"
    agreement = int((max(spam_votes, ham_votes) / 4) * 100)
    return consensus, agreement
