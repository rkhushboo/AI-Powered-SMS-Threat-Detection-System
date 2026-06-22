import csv
import io
from collections import Counter

import plotly.graph_objects as go
import streamlit as st
from visualization import (
    spam_ham_pie,
    model_metrics_bar,
    placeholder_confusion_matrix,
    placeholder_roc_curve,
    placeholder_pr_curve,
    placeholder_wordcloud,
    performance_bar_chart,
    radar_comparison_chart,
    analytics_summary_cards,
    analytics_insight_cards,
    architecture_summary_cards,
    model_ranking_order,
    best_overall_model,
    top_metric_models,
    MODEL_ANALYTICS_METRICS,
    load_sms_dataset,
    eda_summary_stats,
    class_distribution_bar,
    class_distribution_pie,
    message_length_histogram,
    message_length_boxplot,
    word_count_distribution,
    average_word_count_summary,
    top_common_words,
    common_words_bar_chart,
    generate_wordcloud_image,
    ngram_chart,
    vocabulary_statistics,
    eda_insight_cards,
)
from config import (
    APP_TITLE,
    APP_SUBTITLE,
    MODEL_NAMES,
    THEME_COLORS,
)
from prediction import compute_consensus, predict_all_models


def _set_page_config():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _inject_css():
    css = f"""
    <style>
    :root {{
        --bg: #0F172A; /* updated background */
        --sidebar-bg: #0a1220;
        --card-bg: rgba(255,255,255,0.03);
        --glass-border: rgba(255,255,255,0.06);
        --primary: {THEME_COLORS['primary']};
        --secondary: {THEME_COLORS['secondary'] if 'secondary' in THEME_COLORS else THEME_COLORS['primary']};
        --accent: {THEME_COLORS['accent']};
        --muted: rgba(255,255,255,0.65);
        --spam-red: #ff4d6d;
        --ham-green: #22c55e;
    }}
    html, body, [class*="stApp"] {{
        background: var(--bg) !important;
        color: white;
        font-family: Inter, Roboto, -apple-system, 'Segoe UI', sans-serif;
    }}
    /* Sidebar */
    .css-1d391kg {{
        background: var(--sidebar-bg) !important;
    }}
    .glass-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }}
    .kpi {{
        padding: 12px 16px;
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.03);
        text-align: left;
    }}
    .kpi .label {{ color: var(--muted); font-size:12px }}
    .kpi .value {{ font-size:20px; font-weight:700 }}
    .model-card {{ padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.04); background:rgba(255,255,255,0.015); }}
    .consensus-card {{ padding:18px; border-radius:14px; background: linear-gradient(90deg,var(--primary),var(--accent)); color:#021; font-weight:700; text-align:center }}
    .small-muted {{ color:var(--muted); font-size:12px }}
    .prediction-badge {{ padding:6px 10px; border-radius:999px; font-weight:700; color:#021 }}
    .prediction-spam {{ background:var(--spam-red); color:white }}
    .prediction-ham {{ background:var(--ham-green); color:white }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def sidebar_navigation():
    st.sidebar.markdown("""
    <div style='display:flex;align-items:center;gap:12px'>
      <div style='width:42px;height:42px;border-radius:8px;background:var(--primary);display:flex;align-items:center;justify-content:center;font-weight:700'>AI</div>
      <div>
        <div style='font-weight:700'>{}</div>
        <div style='font-size:12px;color:var(--muted)'>{}</div>
      </div>
    </div>
    """.format(APP_TITLE, APP_SUBTITLE), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    nav = st.sidebar.radio(
        "",
        (
            "🏠 Dashboard",
            "⚡ Live Prediction",
            "📤 Batch Prediction",
            "🏆 Model Comparison",
            "📈 Model Analytics",
            "🔎 EDA",
            "ℹ️ About Project",
        ),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models (deployed locally)**")
    for m in MODEL_NAMES:
        st.sidebar.markdown(f":small_blue_diamond: {m}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:12px;color:var(--muted)'>NLP • Deep Learning • TensorFlow • Batch scoring</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:12px;color:var(--muted)'>Local model inference with TensorFlow/Keras artifacts.</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:12px;color:var(--muted)'>Upload CSV files and export consensus predictions.</div>", unsafe_allow_html=True)
    return nav


def kpi_row():
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
    # Placeholder KPIs
    col1.markdown("<div class='kpi glass-card'><b>Total Messages</b><div style='font-size:20px'>5,572</div></div>", unsafe_allow_html=True)
    col2.markdown("<div class='kpi glass-card'><b>Spam</b><div style='font-size:20px'>747</div></div>", unsafe_allow_html=True)
    col3.markdown("<div class='kpi glass-card'><b>Ham</b><div style='font-size:20px'>4,825</div></div>", unsafe_allow_html=True)
    col4.markdown("<div class='kpi glass-card'><b>Vocab Size</b><div style='font-size:20px'>4,788</div></div>", unsafe_allow_html=True)
    col5.markdown("<div class='kpi glass-card'><b>Best Accuracy</b><div style='font-size:20px'>99.10%</div></div>", unsafe_allow_html=True)
    col6.markdown("<div class='kpi glass-card'><b>Best F1 Score</b><div style='font-size:20px'>96.73%</div></div>", unsafe_allow_html=True)


def render_dashboard():
    # Hero banner
    st.markdown("<div class='glass-card' style='display:flex;justify-content:space-between;align-items:center'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"<div style='padding:6px'><h2 style='margin:0'>{APP_TITLE}</h2><div class='small-muted'>{APP_SUBTITLE}</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted' style='margin-top:8px'>Detect spam messages using Machine Learning, Deep Learning, and Transformer-based AI models.</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align:right'><div class='kpi'><div class='label'>Models</div><div class='value'>4</div></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # KPI row
    kpi_row()

    # Main charts
    left, right = st.columns([2,1])
    with left:
        fig = spam_ham_pie()
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = model_metrics_bar()
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='glass-card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.markdown("#### Architecture & Prediction Pipeline")
    st.markdown("A parallel inference pipeline runs TextCNN, BiLSTM, BiGRU and ANN models to deliver a consensus spam score.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_live_prediction():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Live Prediction — Real-time SMS Classification")
    st.markdown("Enter an SMS message on the left and view per-model predictions on the right.")
    st.markdown("</div>", unsafe_allow_html=True)

    if "sms_input" not in st.session_state:
        st.session_state["sms_input"] = ""

    if "model_results" not in st.session_state:
        st.session_state["model_results"] = None

    textarea, actions = st.columns([3,1])
    with textarea:
        st.session_state["sms_input"] = st.text_area(
            "Enter SMS message",
            value=st.session_state["sms_input"],
            height=180,
            placeholder="Enter or paste an SMS message here...",
            key="sms_text",
        )
        st.selectbox(
            "Example messages",
            [
                "Free entry in 2 a wkly comp to win",
                "Hey, are we meeting today?",
                "Urgent! Your account will be closed.",
            ],
            key="example_messages",
        )

    with actions:
        st.write("")
        if st.button("Predict", key="live_predict"):
            try:
                model_results = predict_all_models(st.session_state["sms_input"])
                st.session_state["model_results"] = model_results
            except Exception as exc:
                st.error(f"Model inference failed: {exc}")
                st.session_state["model_results"] = None

        if st.button("Clear", key="live_clear"):
            st.session_state["sms_input"] = ""
            st.session_state["model_results"] = None

    model_results = st.session_state.get("model_results")
    grid = []
    for model_name in ["TextCNN", "BiLSTM", "BiGRU", "ANN"]:
        if model_results and model_name in model_results:
            res = model_results[model_name]
            status = res["prediction"] == "SPAM"
            grid.append(
                {
                    "name": model_name,
                    "prediction": res["prediction"],
                    "probability": f"{res['probability']*100:.1f}%",
                    "confidence": res["confidence_level"],
                    "time": f"{res['inference_time']} ms",
                    "status": "spam" if status else "ham",
                    "confidence_score": res["confidence_score"],
                }
            )
        else:
            grid.append(
                {
                    "name": model_name,
                    "prediction": "Pending",
                    "probability": "--",
                    "confidence": "Pending",
                    "time": "--",
                    "status": "ham",
                    "confidence_score": 0.0,
                }
            )

    if model_results:
        consensus, agreement = compute_consensus(model_results)
        st.markdown("<div class='glass-card' style='margin-top:14px'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='consensus-card'>Consensus Verdict: {consensus} • Model Agreement: {agreement}%</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    valid_votes = [model for model in grid if model["prediction"] in {"SPAM", "HAM"}]
    spam_votes = sum(1 for model in valid_votes if model["prediction"] == "SPAM")
    ham_votes = sum(1 for model in valid_votes if model["prediction"] == "HAM")
    total_votes = len(valid_votes)
    agreement_score = int((max(spam_votes, ham_votes) / total_votes) * 100) if total_votes else 0
    if total_votes == 0:
        consensus = "Pending"
    elif spam_votes == ham_votes:
        consensus = "TIED"
    else:
        consensus = "SPAM" if spam_votes > ham_votes else "HAM"
    pending_votes = sum(1 for model in grid if model["prediction"] == "Pending")

    rows = [st.columns(2), st.columns(2)]
    idx = 0
    for row_cols in rows:
        for col in row_cols:
            card = grid[idx]
            status_class = "prediction-spam" if card["status"] == "spam" else "prediction-ham"
            col.markdown(
                f"<div class='model-card glass-card'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'><div><b>{card['name']}</b></div><div class='small-muted'>{card['time']}</div></div>"
                f"<div style='margin-top:12px'><span class='prediction-badge {status_class}'>{card['prediction']}</span></div>"
                f"<div style='margin-top:10px'><strong>Probability:</strong> {card['probability']}</div>"
                f"<div class='small-muted'>Confidence Level: {card['confidence']}</div>"
                f"<div style='margin-top:10px'><div style='background:rgba(255,255,255,0.08);border-radius:8px;height:10px;overflow:hidden'><div style='width:{card['confidence_score']*100 if card['confidence_score'] else 0}%;height:10px;background:linear-gradient(90deg,var(--primary),var(--accent));'></div></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            idx += 1

    st.markdown("<div class='glass-card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='consensus-card'>Models Voting Spam: {spam_votes} / {total_votes if total_votes else 4} — Models Voting Ham: {ham_votes} / {total_votes if total_votes else 4}<br>Pending: {pending_votes}<br>Final Verdict: {consensus} — Agreement: {agreement_score}%</div>"
    , unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if model_results:
        with st.expander("Debug: Model Results and Agreement", expanded=True):
            for model_name, result in model_results.items():
                error_text = f" — Error: {result['error']}" if result.get("error") else ""
                st.write(f"**{model_name}** — Prediction: {result['prediction']}, Probability: {result['probability']:.4f}, Confidence: {result['confidence_level']}{error_text}")
            st.write(f"Agreement score: {agreement_score}%")
            st.write(f"Spam votes: {spam_votes}, Ham votes: {ham_votes}")
            st.write(f"Pending predictions: {pending_votes}")


def render_batch_prediction():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Batch Prediction — Upload SMS CSV")
    st.markdown("Upload a CSV file with a `message` column. All deployed models will score each record and produce consensus predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")
    sample_csv = "message\nHello world\nFree entry to win"
    st.download_button("Download Sample CSV", data=sample_csv, file_name="sample_sms_batch.csv", mime="text/csv")

    if not uploaded_file:
        st.info("Upload a CSV file containing a 'message' column to begin batch inference.")
        return

    file_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(file_content))
    if not reader.fieldnames:
        st.error("Uploaded file is empty or not a valid CSV.")
        return

    normalized_fields = {name.strip().lower(): name for name in reader.fieldnames}
    if "message" not in normalized_fields:
        st.error("Required column 'message' not found. Please upload a CSV with a 'message' column.")
        return

    message_field = normalized_fields["message"]
    rows = [row for row in reader]
    if not rows:
        st.error("The uploaded file contains no rows.")
        return

    messages = []
    empty_rows = []
    for idx, row in enumerate(rows, start=1):
        msg = (row.get(message_field, "") or "").strip()
        if msg:
            messages.append(msg)
        else:
            empty_rows.append(idx)

    if empty_rows:
        st.error(f"Found empty messages in rows: {', '.join(map(str, empty_rows))}. Please remove or fill them.")
        return

    if st.button("Run Batch Prediction", key="run_batch"):
        total = len(messages)
        progress = st.progress(0)
        status_text = st.empty()
        predictions = []

        with st.spinner("Processing batch predictions..."):
            for index, message in enumerate(messages, start=1):
                model_results = predict_all_models(message)
                consensus, agreement = compute_consensus(model_results)
                predictions.append(
                    {
                        "Original_Message": message,
                        "TextCNN_Prediction": model_results["TextCNN"]["prediction"],
                        "TextCNN_Probability": f"{model_results['TextCNN']['probability']:.4f}",
                        "BiLSTM_Prediction": model_results["BiLSTM"]["prediction"],
                        "BiLSTM_Probability": f"{model_results['BiLSTM']['probability']:.4f}",
                        "BiGRU_Prediction": model_results["BiGRU"]["prediction"],
                        "BiGRU_Probability": f"{model_results['BiGRU']['probability']:.4f}",
                        "ANN_Prediction": model_results["ANN"]["prediction"],
                        "ANN_Probability": f"{model_results['ANN']['probability']:.4f}",
                        "Consensus_Prediction": consensus,
                        "Agreement_Percentage": agreement,
                    }
                )
                progress.progress(int(index / total * 100))
                status_text.text(f"Processed {index} of {total} messages.")

        progress.empty()
        status_text.success("Batch processing complete.")

        spam_count = sum(1 for row in predictions if row["Consensus_Prediction"] == "SPAM")
        ham_count = sum(1 for row in predictions if row["Consensus_Prediction"] == "HAM")
        spam_pct = round(spam_count / total * 100, 1) if total else 0.0
        ham_pct = round(ham_count / total * 100, 1) if total else 0.0
        avg_agreement = round(sum(row["Agreement_Percentage"] for row in predictions) / total, 1)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Batch Summary")
        summary_cols = st.columns(6)
        summary_cols[0].markdown(f"<div class='kpi'><div class='label'>Total Messages</div><div class='value'>{total}</div></div>", unsafe_allow_html=True)
        summary_cols[1].markdown(f"<div class='kpi'><div class='label'>Spam Messages</div><div class='value'>{spam_count}</div></div>", unsafe_allow_html=True)
        summary_cols[2].markdown(f"<div class='kpi'><div class='label'>Ham Messages</div><div class='value'>{ham_count}</div></div>", unsafe_allow_html=True)
        summary_cols[3].markdown(f"<div class='kpi'><div class='label'>Spam Percentage</div><div class='value'>{spam_pct}%</div></div>", unsafe_allow_html=True)
        summary_cols[4].markdown(f"<div class='kpi'><div class='label'>Ham Percentage</div><div class='value'>{ham_pct}%</div></div>", unsafe_allow_html=True)
        summary_cols[5].markdown(f"<div class='kpi'><div class='label'>Avg Agreement</div><div class='value'>{avg_agreement}%</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        consensus_counts = Counter(row["Consensus_Prediction"] for row in predictions)
        agreement_values = [row["Agreement_Percentage"] for row in predictions]
        model_counts = {model: Counter(row[f"{model}_Prediction"] for row in predictions) for model in ["TextCNN", "BiLSTM", "BiGRU", "ANN"]}

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Prediction Visualizations")
        viz1, viz2 = st.columns(2)
        with viz1:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(consensus_counts.keys()),
                        y=list(consensus_counts.values()),
                        marker_color=["#ff4d6d" if label == "SPAM" else "#22c55e" for label in consensus_counts.keys()],
                    )
                ]
            )
            fig.update_layout(title="Consensus Prediction Distribution", xaxis_title="Prediction", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        with viz2:
            fig = go.Figure(
                data=[
                    go.Histogram(x=agreement_values, nbinsx=10, marker_color="#00b8ff")
                ]
            )
            fig.update_layout(title="Model Agreement Distribution", xaxis_title="Agreement (%)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        viz3, viz4 = st.columns(2)
        with viz3:
            fig = go.Figure(
                data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=spam_pct,
                        title={"text": "Spam Percentage"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ff4d6d"}},
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        with viz4:
            fig = go.Figure()
            for model_name, counts in model_counts.items():
                fig.add_trace(
                    go.Bar(
                        x=[model_name, model_name],
                        y=[counts.get("HAM", 0), counts.get("SPAM", 0)],
                        name=model_name,
                    )
                )
            fig.update_layout(barmode="group", title="Prediction Comparison Across Models", xaxis_title="Model", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        data_buffer = io.StringIO()
        writer = csv.DictWriter(data_buffer, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
        st.download_button(
            "Download Results CSV",
            data=data_buffer.getvalue().encode("utf-8"),
            file_name="sms_predictions.csv",
            mime="text/csv",
        )

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<details><summary style='font-size:16px;color:#fff;cursor:pointer;padding:8px'>Batch Insights</summary>")
        pairwise_insights = []
        pairs = [("TextCNN", "BiLSTM"), ("TextCNN", "BiGRU"), ("TextCNN", "ANN"), ("BiLSTM", "BiGRU"), ("BiLSTM", "ANN"), ("BiGRU", "ANN")]
        for a, b in pairs:
            agreement_pct = round(sum(1 for row in predictions if row[f"{a}_Prediction"] == row[f"{b}_Prediction"]) / total * 100, 1)
            pairwise_insights.append(f"{a} and {b} agreed on {agreement_pct}% of records.")
        st.markdown(f"- {spam_pct}% of uploaded messages were classified as spam.")
        st.markdown(f"- Average model agreement was {avg_agreement}%.")
        for insight in pairwise_insights[:3]:
            st.markdown(f"- {insight}")
        st.markdown("</details>")
        st.markdown("</div>", unsafe_allow_html=True)

        st.dataframe(predictions[:10], use_container_width=True)


def render_model_comparison():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Model Comparison")
    st.markdown("Compare deployed model performance across accuracy, precision, recall, F1 and ROC-AUC metrics.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(model_metrics_bar(), use_container_width=True)
    st.markdown("<div class='glass-card'>This comparison uses the current SMS spam classification ensemble and highlights the strongest model candidates for production deployment.</div>", unsafe_allow_html=True)


def render_model_analytics():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Model Analytics Dashboard")
    st.markdown("A polished analytics section showcasing model ranking, metrics, and performance insights.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 1: Model Leaderboard
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Model Leaderboard")
    leaderboard = model_ranking_order()
    cols = st.columns(4)
    for idx, model_name in enumerate(leaderboard, start=1):
        metric = performance_bar_chart("F1")  # placeholder for consistent color logic
        best_mark = " — Best" if idx == 1 else ""
        cols[idx - 1].markdown(
            f"<div class='glass-card' style='padding:18px; background:rgba(255,255,255,0.04);'>"
            f"<div style='font-size:18px;font-weight:700;margin-bottom:8px'>{idx}. {model_name}{best_mark}</div>"
            f"<div style='font-size:14px;color:rgba(255,255,255,0.7);'>F1: {MODEL_ANALYTICS_METRICS[model_name]['F1']*100:.2f}%</div>"
            f"<div style='font-size:14px;color:rgba(255,255,255,0.7);margin-top:6px'>ROC-AUC: {MODEL_ANALYTICS_METRICS[model_name]['ROC-AUC']*100:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: KPI cards
    summary = analytics_summary_cards()
    kpi_cols = st.columns(4)
    for idx, (label, value) in enumerate(summary.items()):
        kpi_cols[idx].markdown(
            f"<div class='glass-card' style='padding:20px;text-align:center'>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.7);text-transform:uppercase;margin-bottom:8px'>{label}</div>"
            f"<div style='font-size:24px;font-weight:700'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Section 3: Interactive metric comparisons
    st.markdown("<div class='glass-card' style='padding:18px'>", unsafe_allow_html=True)
    st.markdown("#### Metric Comparison Charts")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for metric in metrics:
        st.plotly_chart(performance_bar_chart(metric), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 4: Radar Chart
    st.markdown("<div class='glass-card' style='padding:18px'>", unsafe_allow_html=True)
    st.markdown("#### Multi-Metric Radar Comparison")
    st.plotly_chart(radar_comparison_chart(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 5: Ranking analysis cards
    top_metrics = top_metric_models()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Model Ranking Analysis")
    rank_cols = st.columns(4)
    rank_labels = [
        ("Best Overall Model", best_overall_model()),
        ("Highest Precision", top_metrics['Highest Precision']),
        ("Highest Recall", top_metrics['Highest Recall']),
        ("Highest F1", top_metrics['Highest F1']),
    ]
    for idx, (label, model_name) in enumerate(rank_labels):
        rank_cols[idx].markdown(
            f"<div class='glass-card' style='padding:18px;text-align:center'>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:20px;font-weight:700'>{model_name}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 6: Performance insights
    notes = analytics_insight_cards()
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Performance Insights")
    for note in notes:
        st.markdown(f"- {note}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 7: Architecture summary
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Model Architecture Summary")
    arch_cards = architecture_summary_cards()
    arch_cols = st.columns(4)
    for idx, card in enumerate(arch_cards):
        arch_cols[idx].markdown(
            f"<div class='glass-card' style='padding:18px;background:rgba(255,255,255,0.02);'>"
            f"<div style='font-size:16px;font-weight:700;color:{card['color']};margin-bottom:8px'>{card['name']}</div>"
            f"<div style='font-size:14px;color:rgba(255,255,255,0.75);line-height:1.5'>{card['description']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_eda():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Exploratory Data Analysis (EDA)")
    st.markdown("Professional dataset dashboard powered by the SMS Spam Collection dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

    rows = load_sms_dataset()
    stats = eda_summary_stats(rows)
    ham_msgs = [r['Message'] for r in rows if r['Category'] == 'ham']
    spam_msgs = [r['Message'] for r in rows if r['Category'] == 'spam']
    insights = eda_insight_cards()

    # Section 1: Dataset Overview
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Dataset Overview")
    kpi_cols = st.columns(5)
    for idx, (label, value) in enumerate(stats.items()):
        kpi_cols[idx].markdown(
            f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: Class Distribution
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Class Distribution")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.plotly_chart(class_distribution_bar(rows), use_container_width=True)
    with dist_col2:
        st.plotly_chart(class_distribution_pie(rows), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3: Message Length Analysis
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Message Length Analysis")
    len_col1, len_col2 = st.columns(2)
    with len_col1:
        st.plotly_chart(message_length_histogram(rows), use_container_width=True)
    with len_col2:
        st.plotly_chart(message_length_boxplot(rows), use_container_width=True)
    st.markdown("<div style='color:rgba(255,255,255,0.8);margin-top:10px'>Spam messages show more length variation and generally longer text than ham messages.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 4: Word Count Analysis
    summary_counts = average_word_count_summary(rows)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Word Count Analysis")
    wc_col1, wc_col2 = st.columns([1,2])
    with wc_col1:
        st.markdown(
            f"<div class='glass-card' style='background:rgba(255,255,255,0.02);padding:18px'>"
            f"<div style='font-weight:700;margin-bottom:6px'>Average Word Count</div>"
            f"<div style='font-size:20px'>{summary_counts['Average Ham Words']} (Ham)</div>"
            f"<div style='font-size:20px'>{summary_counts['Average Spam Words']} (Spam)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with wc_col2:
        st.plotly_chart(word_count_distribution(rows), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 5: Most Common Words
    spam_words = top_common_words(rows, 'spam')
    ham_words = top_common_words(rows, 'ham')
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Most Common Words")
    word_col1, word_col2 = st.columns(2)
    with word_col1:
        st.plotly_chart(common_words_bar_chart(spam_words, 'Top 20 Spam Words'), use_container_width=True)
    with word_col2:
        st.plotly_chart(common_words_bar_chart(ham_words, 'Top 20 Ham Words'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 6: Word Clouds
    spam_text = ' '.join(spam_msgs)
    ham_text = ' '.join(ham_msgs)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Word Clouds")
    cloud_col1, cloud_col2 = st.columns(2)
    with cloud_col1:
        st.image(generate_wordcloud_image(spam_text), caption='Spam Word Cloud', use_column_width=True)
    with cloud_col2:
        st.image(generate_wordcloud_image(ham_text), caption='Ham Word Cloud', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 7: N-Gram Analysis
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### N-Gram Analysis")
    ngram_col1, ngram_col2 = st.columns(2)
    with ngram_col1:
        st.plotly_chart(ngram_chart(rows, 'spam', 2, 'Top 15 Spam Bigrams'), use_container_width=True)
        st.plotly_chart(ngram_chart(rows, 'spam', 3, 'Top 15 Spam Trigrams'), use_container_width=True)
    with ngram_col2:
        st.plotly_chart(ngram_chart(rows, 'ham', 2, 'Top 15 Ham Bigrams'), use_container_width=True)
        st.plotly_chart(ngram_chart(rows, 'ham', 3, 'Top 15 Ham Trigrams'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 8: Vocabulary Statistics
    vocab_stats = vocabulary_statistics(rows)
    stat_cols = st.columns(6)
    for idx, (label, value) in enumerate(vocab_stats.items()):
        stat_cols[idx].markdown(
            f"<div class='glass-card' style='padding:16px;text-align:center'>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.7);text-transform:uppercase;margin-bottom:8px'>{label}</div>"
            f"<div style='font-size:20px;font-weight:700'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 9: Key Insights
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Key Insights")
    for note in insights:
        st.markdown(f"- {note}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_about():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### About Project")
    st.markdown("Project motivation, preprocessing pipeline, model summaries and deployment notes.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Project Overview**")
    st.write("AI-Powered SMS Threat Detection System — a production-style UI for multi-model SMS spam classification.")

    st.markdown("**Dataset Information**")
    st.write("SMS Spam Collection Dataset — 5,572 messages (747 spam, 4,825 ham)")

    st.markdown("**Preprocessing Pipeline**")
    st.markdown("1. Normalize text to lowercase  \n2. Remove special characters and punctuation  \n3. Remove stopwords  \n4. Tokenize text  \n5. Convert to sequences using a saved Keras tokenizer  \n6. Pad sequences to the configured max length")

    st.markdown("**Model Architecture Summary**")
    st.write("TextCNN, BiLSTM, BiGRU and ANN models are used for ensemble inference, with a consensus voting layer for reliable spam detection.")

    st.markdown("**Technology Stack**")
    st.write("Streamlit, TensorFlow/Keras, Plotly, Python, and local CSV batch processing")

    st.markdown("**Deployment Notes**")
    st.write("Ensure model artifacts and tokenizer files are available in the repository root before launching the app.")

    st.markdown("**Future Enhancements**")
    st.write("Explainability, model monitoring, A/B testing, multi-lingual support, production model serving")


def main():
    _set_page_config()
    _inject_css()

    nav = sidebar_navigation()

    st.header("")

    # Normalize navigation (strip emoji)
    if "Dashboard" in nav:
        render_dashboard()
    elif "Live Prediction" in nav:
        render_live_prediction()
    elif "Batch Prediction" in nav:
        render_batch_prediction()
    elif "Model Comparison" in nav:
        render_model_comparison()
    elif "Model Analytics" in nav:
        render_model_analytics()
    elif "EDA" in nav:
        render_eda()
    elif "About" in nav:
        render_about()

    st.markdown(
        "<div style='text-align:center;padding:14px;color:var(--muted);font-size:12px'>"
        "Built with Streamlit · Local TensorFlow inference · SMS spam detection ensemble</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
