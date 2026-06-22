import csv
from collections import Counter
from itertools import islice
from pathlib import Path

import plotly.graph_objects as go
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from config import VOCAB_SIZE
from preprocessing import clean_text, tokenize_text


def spam_ham_pie():
    labels = ["Ham", "Spam"]
    values = [4825, 747]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45)])
    fig.update_traces(marker=dict(colors=['#22c55e','#ff4d6d']), hoverinfo='label+percent+value')
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def model_metrics_bar():
    models = ["LightGBM", "TextCNN", "BiLSTM", "DistilBERT"]
    accuracy = [0.985, 0.991, 0.988, 0.989]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=[a*100 for a in accuracy], marker_color=['#00b8ff','#8a5cff','#00ffd1','#8a5cff']))
    fig.update_layout(yaxis_title='Accuracy (%)', margin=dict(l=10, r=10, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def placeholder_confusion_matrix():
    z = [[900, 20], [15, 65]]
    fig = go.Figure(data=go.Heatmap(z=z, x=['Pred Ham', 'Pred Spam'], y=['True Ham', 'True Spam'], colorscale='Blues'))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def placeholder_roc_curve():
    x = np.linspace(0, 1, 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.sqrt(x), mode='lines', name='TextCNN', line=dict(color='#00b8ff')))
    fig.add_trace(go.Scatter(x=x, y=np.power(x, 0.8), mode='lines', name='BiLSTM', line=dict(color='#8a5cff')))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', margin=dict(l=10, r=10, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def placeholder_pr_curve():
    x = np.linspace(0, 1, 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=1 - x**0.5, mode='lines', name='TextCNN', line=dict(color='#00b8ff')))
    fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', margin=dict(l=10, r=10, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def placeholder_wordcloud(width=800, height=400):
    img = Image.new('RGB', (width, height), color=(11,16,32))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype('arial.ttf', 36)
    except Exception:
        f = ImageFont.load_default()
    text = "Word Cloud Placeholder"
    w, h = d.textsize(text, font=f)
    d.text(((width-w)/2, (height-h)/2), text, font=f, fill=(0,184,255))
    return img


DATA_PATH = Path(__file__).resolve().parent / "data" / "SPAM text message 20170820 - Data.csv"

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None


@st.cache_data(show_spinner=False)
def load_sms_dataset():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "Category": row.get("Category", "").strip().lower(),
                    "Message": row.get("Message", "").strip(),
                }
            )
    return rows


def _split_by_category(rows):
    ham_msgs = [r["Message"] for r in rows if r["Category"] == "ham"]
    spam_msgs = [r["Message"] for r in rows if r["Category"] == "spam"]
    return ham_msgs, spam_msgs


def _message_lengths(rows):
    return [len(r["Message"]) for r in rows]


def _word_counts(rows):
    return [len(tokenize_text(r["Message"])) for r in rows]


def _category_word_counter(rows, category):
    counter = Counter()
    for r in rows:
        if r["Category"] == category:
            counter.update(tokenize_text(r["Message"]))
    return counter


def _top_ngrams(rows, category, n, top_n=15):
    counter = Counter()
    for r in rows:
        if r["Category"] == category:
            tokens = tokenize_text(r["Message"])
            counter.update(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
    return [(" ".join(ngram), count) for ngram, count in counter.most_common(top_n)]


def eda_summary_stats(rows):
    total = len(rows)
    ham = sum(1 for r in rows if r["Category"] == "ham")
    spam = total - ham
    spam_pct = round((spam / total) * 100, 1) if total else 0.0
    return {
        "Total Messages": total,
        "Ham Messages": ham,
        "Spam Messages": spam,
        "Spam Percentage": f"{spam_pct}%",
        "Vocabulary Size": VOCAB_SIZE,
    }


def class_distribution_bar(rows):
    labels = ["Ham", "Spam"]
    values = [sum(1 for r in rows if r["Category"] == "ham"), sum(1 for r in rows if r["Category"] == "spam")]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#22c55e", "#ff4d6d"],
                text=[f"{v}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(title="Class Distribution", yaxis_title="Count", xaxis_title="Category")
    return _analytics_base_layout(fig)


def class_distribution_pie(rows):
    labels = ["Ham", "Spam"]
    values = [sum(1 for r in rows if r["Category"] == "ham"), sum(1 for r in rows if r["Category"] == "spam")]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="percent+label")])
    fig.update_traces(marker=dict(colors=["#22c55e", "#ff4d6d"]))
    fig.update_layout(title="Ham vs Spam Share")
    return _analytics_base_layout(fig)


def message_length_histogram(rows):
    ham_msgs, spam_msgs = _split_by_category(rows)
    ham_lengths = [len(m) for m in ham_msgs]
    spam_lengths = [len(m) for m in spam_msgs]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ham_lengths, name="Ham", opacity=0.75, marker_color="#22c55e", nbinsx=40))
    fig.add_trace(go.Histogram(x=spam_lengths, name="Spam", opacity=0.75, marker_color="#ff4d6d", nbinsx=40))
    fig.update_layout(barmode="overlay", title="Message Length Distribution", xaxis_title="Message Length (chars)", yaxis_title="Count")
    return _analytics_base_layout(fig)


def message_length_boxplot(rows):
    ham_msgs, spam_msgs = _split_by_category(rows)
    fig = go.Figure()
    fig.add_trace(go.Box(y=[len(m) for m in ham_msgs], name="Ham", marker_color="#22c55e"))
    fig.add_trace(go.Box(y=[len(m) for m in spam_msgs], name="Spam", marker_color="#ff4d6d"))
    fig.update_layout(title="Message Length Boxplot", yaxis_title="Length (chars)")
    return _analytics_base_layout(fig)


def word_count_distribution(rows):
    ham_msgs, spam_msgs = _split_by_category(rows)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[len(tokenize_text(m)) for m in ham_msgs], name="Ham", opacity=0.75, marker_color="#22c55e", nbinsx=20))
    fig.add_trace(go.Histogram(x=[len(tokenize_text(m)) for m in spam_msgs], name="Spam", opacity=0.75, marker_color="#ff4d6d", nbinsx=20))
    fig.update_layout(barmode="overlay", title="Word Count Distribution", xaxis_title="Word Count", yaxis_title="Count")
    return _analytics_base_layout(fig)


def average_word_count_summary(rows):
    ham_msgs, spam_msgs = _split_by_category(rows)
    ham_avg = round(sum(len(tokenize_text(m)) for m in ham_msgs) / len(ham_msgs), 2) if ham_msgs else 0
    spam_avg = round(sum(len(tokenize_text(m)) for m in spam_msgs) / len(spam_msgs), 2) if spam_msgs else 0
    return {
        "Average Ham Words": ham_avg,
        "Average Spam Words": spam_avg,
    }


def top_common_words(rows, category, top_n=20):
    counter = _category_word_counter(rows, category)
    return counter.most_common(top_n)


def common_words_bar_chart(words, title):
    labels, values = zip(*words) if words else ([], [])
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(values)[::-1],
                y=list(labels)[::-1],
                orientation="h",
                marker_color="#00b8ff",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="Count", yaxis_title="Word")
    return _analytics_base_layout(fig)


def generate_wordcloud_image(text, width=480, height=360):
    if WordCloud is None or not text.strip():
        return placeholder_wordcloud(width, height)
    cloud = WordCloud(
        width=width,
        height=height,
        background_color="#0F172A",
        colormap="plasma",
        stopwords=set(),
        contour_width=0,
    ).generate(text)
    return cloud.to_image()


def ngram_chart(rows, category, n, title):
    top_ngrams = _top_ngrams(rows, category, n)
    labels, counts = zip(*top_ngrams) if top_ngrams else ([], [])
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(counts)[::-1],
                y=list(labels)[::-1],
                orientation="h",
                marker_color="#8a5cff",
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="Count", yaxis_title=f"Top {n}-grams")
    return _analytics_base_layout(fig)


def vocabulary_statistics(rows):
    lengths = _message_lengths(rows)
    return {
        "Vocabulary Size": VOCAB_SIZE,
        "Average Message Length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "Maximum Message Length": max(lengths) if lengths else 0,
        "Minimum Message Length": min(lengths) if lengths else 0,
        "95th Percentile Length": int(np.percentile(lengths, 95)) if lengths else 0,
        "99th Percentile Length": int(np.percentile(lengths, 99)) if lengths else 0,
    }


def eda_insight_cards():
    return [
        "The dataset is moderately imbalanced with around 13% spam messages.",
        "Spam messages tend to be longer and more promotional than ham messages.",
        "Promotional terms like 'free', 'win', and 'text' dominate the spam vocabulary.",
        "Ham messages contain more conversational language and shorter, personal phrases.",
    ]


MODEL_ANALYTICS_METRICS = {
    "TextCNN": {
        "Accuracy": 0.991039,
        "Precision": 0.948718,
        "Recall": 0.986667,
        "F1": 0.967320,
        "ROC-AUC": 0.997101,
    },
    "BiLSTM": {
        "Accuracy": 0.991039,
        "Precision": 0.986111,
        "Recall": 0.946667,
        "F1": 0.965986,
        "ROC-AUC": 0.994010,
    },
    "BiGRU": {
        "Accuracy": 0.985663,
        "Precision": 0.971831,
        "Recall": 0.920000,
        "F1": 0.945205,
        "ROC-AUC": 0.992298,
    },
    "ANN": {
        "Accuracy": 0.985663,
        "Precision": 0.985507,
        "Recall": 0.906667,
        "F1": 0.944444,
        "ROC-AUC": 0.992464,
    },
}

MODEL_ANALYTICS_ORDER = ["TextCNN", "BiLSTM", "BiGRU", "ANN"]
MODEL_ANALYTICS_COLOR = {
    "TextCNN": "#00b8ff",
    "BiLSTM": "#8a5cff",
    "BiGRU": "#00ffd1",
    "ANN": "#f97316",
}


def _analytics_base_layout(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter, Roboto, sans-serif'),
        margin=dict(l=10, r=10, t=36, b=16),
        legend=dict(bgcolor='rgba(255,255,255,0.04)', bordercolor='rgba(255,255,255,0.08)', borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False, ticksuffix='%')
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def performance_bar_chart(metric_key: str):
    models = MODEL_ANALYTICS_ORDER
    values = [MODEL_ANALYTICS_METRICS[m][metric_key] * 100 for m in models]
    colors = [MODEL_ANALYTICS_COLOR[m] for m in models]
    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=models,
                orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.15)', width=1.5)),
                text=[f"{v:.2f}%" for v in values],
                textposition='outside',
            )
        ]
    )
    fig.update_layout(
        title=f"{metric_key} Comparison",
        xaxis=dict(title=f"{metric_key} (%)", range=[min(values) * 0.96, max(values) * 1.08]),
        yaxis=dict(autorange='reversed'),
    )
    return _analytics_base_layout(fig)


def radar_comparison_chart():
    categories = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig = go.Figure()
    for model_name in MODEL_ANALYTICS_ORDER:
        values = [MODEL_ANALYTICS_METRICS[model_name][metric] * 100 for metric in categories]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_name,
                line=dict(color=MODEL_ANALYTICS_COLOR[model_name], width=2),
            )
        )
    fig.update_layout(
        polar=dict(bgcolor='rgba(255,255,255,0.02)', radialaxis=dict(range=[90, 100], tickfont=dict(color='white'))),
        title='Radar Comparison Across Models',
    )
    return _analytics_base_layout(fig)


def analytics_summary_cards():
    best_accuracy = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['Accuracy'])
    best_f1 = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['F1'])
    best_roc = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['ROC-AUC'])
    return {
        'Best Accuracy': f"{MODEL_ANALYTICS_METRICS[best_accuracy]['Accuracy']*100:.2f}% ({best_accuracy})",
        'Best F1 Score': f"{MODEL_ANALYTICS_METRICS[best_f1]['F1']*100:.2f}% ({best_f1})",
        'Best ROC-AUC': f"{MODEL_ANALYTICS_METRICS[best_roc]['ROC-AUC']*100:.2f}% ({best_roc})",
        'Number of Models': len(MODEL_ANALYTICS_METRICS),
    }


def analytics_insight_cards():
    best_roc = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['ROC-AUC'])
    best_precision = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['Precision'])
    best_recall = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['Recall'])
    best_f1 = max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['F1'])
    balanced = max(
        MODEL_ANALYTICS_METRICS,
        key=lambda m: (MODEL_ANALYTICS_METRICS[m]['Recall'] + MODEL_ANALYTICS_METRICS[m]['F1']) / 2,
    )
    return [
        f"{best_roc} achieved the highest ROC-AUC, demonstrating the strongest ranking capability.",
        f"{best_precision} achieved the highest Precision, making it the safest choice for low false positives.",
        f"{balanced} achieved the best balance between Recall and F1, indicating stable deployment readiness.",
        f"{best_f1} achieved the highest F1 score, showing the best overall precision/recall tradeoff.",
    ]


def architecture_summary_cards():
    return [
        {
            'name': 'ANN',
            'description': 'A dense feedforward network with embedding layers, dropout and ReLU activations for efficient SMS classification.',
            'color': MODEL_ANALYTICS_COLOR['ANN'],
        },
        {
            'name': 'BiGRU',
            'description': 'Bidirectional GRU network capturing sequential context with fewer parameters and fast inference.',
            'color': MODEL_ANALYTICS_COLOR['BiGRU'],
        },
        {
            'name': 'BiLSTM',
            'description': 'Bidirectional LSTM architecture modeling long-range dependencies with strong recall performance.',
            'color': MODEL_ANALYTICS_COLOR['BiLSTM'],
        },
        {
            'name': 'TextCNN',
            'description': 'Convolutional text encoder detecting n-gram patterns and delivering top classification accuracy.',
            'color': MODEL_ANALYTICS_COLOR['TextCNN'],
        },
    ]


def model_ranking_order():
    return ["TextCNN", "BiLSTM", "BiGRU", "ANN"]


def top_metric_models():
    return {
        'Highest Precision': max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['Precision']),
        'Highest Recall': max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['Recall']),
        'Highest F1': max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['F1']),
        'Highest ROC-AUC': max(MODEL_ANALYTICS_METRICS, key=lambda m: MODEL_ANALYTICS_METRICS[m]['ROC-AUC']),
    }


def best_overall_model():
    return max(
        MODEL_ANALYTICS_METRICS,
        key=lambda m: (
            MODEL_ANALYTICS_METRICS[m]['Accuracy'] * 0.25
            + MODEL_ANALYTICS_METRICS[m]['Precision'] * 0.25
            + MODEL_ANALYTICS_METRICS[m]['Recall'] * 0.25
            + MODEL_ANALYTICS_METRICS[m]['F1'] * 0.15
            + MODEL_ANALYTICS_METRICS[m]['ROC-AUC'] * 0.10
        ),
    )
