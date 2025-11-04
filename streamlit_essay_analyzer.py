#Instructions
# 1. Create and activate a virtual environment:
#    python3 -m venv venv
#    source venv/bin/activate
#
# 2. Create a requirements.txt file with the content below and install packages:
#    pip install -r requirements.txt
#
# 3. (Optional) Download spaCy model for advanced sentence splitting:
#    python -m spacy download en_core_web_sm
#
# 4. Run the Streamlit app:
#    streamlit run /Users/shivamkulkarni/Desktop/recovered/Essay_Analyzer_App/streamlit_essay_analyzer.py

# requirements.txt content:
# streamlit>=1.0
# matplotlib>=3.0
# numpy>=1.18
# pandas>=1.0
# pillow>=8.0
# reportlab>=3.6
# spacy>=3.0  # Optional

import streamlit as st
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from PIL import Image
import logging

# --- Directory Handling ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
if not os.path.exists(DEFAULT_OUTPUT_DIR):
    os.makedirs(DEFAULT_OUTPUT_DIR)

# --- UI Configuration ---
st.set_page_config(layout="wide")
st.title("Essay Rhythm and Lexical Diversity Analyzer")

st.sidebar.header("Analysis Configuration")
window_size = st.sidebar.number_input("Window size (words)", min_value=20, max_value=500, value=100)
stride = st.sidebar.number_input("Stride (words)", min_value=1, max_value=200, value=20)
hist_bins = st.sidebar.number_input("Histogram bins", min_value=5, max_value=50, value=10)
use_spacy = st.sidebar.checkbox("Use spaCy for sentence splitting (optional)", value=False)
compute_mtld = st.sidebar.checkbox("Compute MTLD (slower)", value=False)
analyze_button = st.sidebar.button("Analyze")

# --- Main App Logic ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Essay Text")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    essay_text = st.text_area("Or paste your essay here", height=400)

    if uploaded_file is not None and not essay_text:
        essay_text = uploaded_file.read().decode("utf-8")
        st.text_area("Or paste your essay here", value=essay_text, height=400)


def tokenize_and_split(text, use_spacy_flag):
    """Splits text into sentences and words."""
    if use_spacy_flag:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
            words = [token.text.lower() for token in doc if token.is_alpha]
            return sentences, words
        except (ImportError, IOError):
            st.error("spaCy or en_core_web_sm not found. Falling back to regex.")
            use_spacy_flag = False

    # Fallback to regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    words = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    return sentences, words


def analyze_text(text, window_size, stride, hist_bins, use_spacy_flag, compute_mtld):
    """Performs the core analysis of the essay."""
    sentences, words = tokenize_and_split(text, use_spacy_flag)
    total_tokens = len(words)

    if total_tokens < 50:
        st.warning("Text has fewer than 50 tokens. Results may be noisy.")
        window_size = min(total_tokens, window_size)

    # Sentence Length Analysis
    sentence_lengths = [len(re.findall(r"\b[a-zA-Z0-9']+\b", s.lower())) for s in sentences]
    mean_len = np.mean(sentence_lengths) if sentence_lengths else 0
    std_len = np.std(sentence_lengths) if sentence_lengths else 0
    cv_len = std_len / mean_len if mean_len > 0 else 0
    
    hist, bin_edges = np.histogram(sentence_lengths, bins=hist_bins)
    probs = hist / hist.sum() if hist.sum() > 0 else []
    entropy = -np.sum(probs * np.log2(probs + 1e-9)) if len(probs) > 0 else 0

    # Lexical Diversity Analysis
    global_ttr = len(set(words)) / total_tokens if total_tokens > 0 else 0
    mtld = None
    if compute_mtld:
        def mtld_calc(tokens):
            if not tokens:
                return 0.0
            ttr_threshold = 0.72
            factors = 0
            factor_tokens = []
            for token in tokens:
                factor_tokens.append(token)
                if len(set(factor_tokens)) / len(factor_tokens) < ttr_threshold:
                    factors += 1
                    factor_tokens = []
            return len(tokens) / factors if factors > 0 else 0.0
        mtld = mtld_calc(words)
    
    centers, uwr_values = [], []
    if total_tokens >= window_size:
        for i in range(0, total_tokens - window_size + 1, stride):
            window = words[i:i + window_size]
            uwr = len(set(window)) / len(window)
            centers.append(i + window_size // 2)
            uwr_values.append(uwr)

    mean_uwr = np.mean(uwr_values) if uwr_values else 0
    std_uwr = np.std(uwr_values) if uwr_values else 0
    threshold = min(0.75, mean_uwr - 0.5 * std_uwr) if mean_uwr > 0 else 0.75

    low_diversity_zones = []
    for i, uwr in enumerate(uwr_values):
        if uwr < threshold:
            start_token = i * stride
            end_token = start_token + window_size
            window_text = " ".join(words[start_token:end_token])
            top_common = pd.Series(words[start_token:end_token]).value_counts().nlargest(5).to_dict()
            low_diversity_zones.append({
                "window_index": i,
                "center_token": centers[i],
                "start_token": start_token,
                "end_token": end_token,
                "uwr": uwr,
                "top_common": top_common,
                "snippet": window_text 
            })

    return {
        "sentence_length": {
            "count_sentences": len(sentences),
            "mean": mean_len,
            "std": std_len,
            "cv": cv_len,
            "entropy": entropy,
            "hist_bins": hist_bins,
            "hist_sample": sentence_lengths[:100]
        },
        "lexical_diversity": {
            "total_tokens": total_tokens,
            "global_ttr": global_ttr,
            "mtld": mtld,
            "window_size": window_size,
            "stride": stride,
            "mean_uwr": mean_uwr,
            "std_uwr": std_uwr,
            "windows": {"centers": centers, "uwr": uwr_values},
            "low_diversity_zones": low_diversity_zones
        }
    }

def create_visuals(analysis_results):
    """Generates and saves plots."""
    # Sentence Length Histogram
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(analysis_results["sentence_length"]["hist_sample"], 
             bins=analysis_results["sentence_length"]["hist_bins"], 
             color="#f9c43b", edgecolor="black")
    ax1.set_title("Sentence Length Distribution")
    ax1.set_xlabel("Sentence Length (words)")
    ax1.set_ylabel("Frequency")
    plt.tight_layout()
    hist_path = os.path.join(DEFAULT_OUTPUT_DIR, "sentence_length_hist.png")
    plt.savefig(hist_path, dpi=150)
    plt.close(fig1)

    # Lexical Diversity Curve
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    lex_div = analysis_results["lexical_diversity"]
    ax2.plot(lex_div["windows"]["centers"], lex_div["windows"]["uwr"], label="UWR")
    low_div_points = [(z["center_token"], z["uwr"]) for z in lex_div["low_diversity_zones"]]
    if low_div_points:
        ax2.scatter(*zip(*low_div_points), color='red', zorder=5, label="Low Diversity")
    ax2.set_ylim(0, 1)
    ax2.set_title("Sliding Window Lexical Diversity")
    ax2.set_xlabel("Token Index (Center of Window)")
    ax2.set_ylabel("Unique Word Ratio (UWR)")
    ax2.legend()
    plt.tight_layout()
    lex_path = os.path.join(DEFAULT_OUTPUT_DIR, "lexical_diversity_curve.png")
    plt.savefig(lex_path, dpi=150)
    plt.close(fig2)

    return hist_path, lex_path

def create_pdf_summary(analysis_results, hist_path, lex_path):
    """Generates a PDF summary of the analysis."""
    pdf_path = os.path.join(DEFAULT_OUTPUT_DIR, "essay_onepager.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Plots
        fig = plt.figure(figsize=(8.27, 11.69)) # A4
        
        # Top plot
        ax1 = fig.add_subplot(2, 1, 1)
        if os.path.exists(hist_path):
            img1 = Image.open(hist_path)
            ax1.imshow(img1)
        ax1.axis('off')

        # Bottom plot
        ax2 = fig.add_subplot(2, 1, 2)
        if os.path.exists(lex_path):
            img2 = Image.open(lex_path)
            ax2.imshow(img2)
        ax2.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Text Summary
        fig = plt.figure(figsize=(8.27, 11.69))
        
        summary_text = "Analysis Summary & Suggestions\n" + "="*30 + "\n"
        sl = analysis_results["sentence_length"]
        ld = analysis_results["lexical_diversity"]
        summary_text += f"Sentences: {sl['count_sentences']}, Mean Length: {sl['mean']:.2f}, Std Dev: {sl['std']:.2f}\n"
        summary_text += f"Tokens: {ld['total_tokens']}, Global TTR: {ld['global_ttr']:.3f}"
        if ld['mtld'] is not None:
            summary_text += f", MTLD: {ld['mtld']:.3f}"
        summary_text += "\n"
        summary_text += f"Mean Window UWR: {ld['mean_uwr']:.3f}, Std Dev: {ld['std_uwr']:.3f}\n\n"
        
        summary_text += "Suggestions:\n" + "-"*20 + "\n"
        if ld["low_diversity_zones"]:
            for i, zone in enumerate(ld["low_diversity_zones"][:6]):
                summary_text += f"{i+1}. Low diversity (UWR: {zone['uwr']:.3f}) near token {zone['center_token']}.\n"
                summary_text += f"   Snippet: ...{zone['snippet'][:100]}...\n"
                summary_text += "   Consider varying word choice or sentence structure.\n\n"
        else:
            summary_text += "No significant low-diversity zones detected.\n"

        fig.text(0.05, 0.95, summary_text, va='top', ha='left', wrap=True, fontsize=8, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path

def create_json_summary(analysis_results, hist_path, lex_path, pdf_path):
    """Generates a JSON summary of the analysis."""
    json_path = os.path.join(DEFAULT_OUTPUT_DIR, "analysis_summary.json")
    analysis_results["generated_paths"] = {
        "hist_png": hist_path,
        "lex_png": lex_path,
        "pdf": pdf_path,
        "json": json_path
    }
    with open(json_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    return json_path


if analyze_button and essay_text:
    with st.spinner("Analyzing..."):
        analysis_results = analyze_text(essay_text, window_size, stride, hist_bins, use_spacy, compute_mtld)
        hist_path, lex_path = create_visuals(analysis_results)
        pdf_path = create_pdf_summary(analysis_results, hist_path, lex_path)
        json_path = create_json_summary(analysis_results, hist_path, lex_path, pdf_path)

        with col2:
            st.subheader("Analysis Results")
            st.image(hist_path, caption="Sentence Length Histogram")
            st.image(lex_path, caption="Lexical Diversity Curve")

            st.subheader("Downloads")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Summary", f, file_name="essay_onepager.pdf")
            with open(json_path, "rb") as f:
                st.download_button("Download JSON Summary", f, file_name="analysis_summary.json")
            with open(hist_path, "rb") as f:
                st.download_button("Download Histogram PNG", f, file_name="sentence_length_hist.png")
            with open(lex_path, "rb") as f:
                st.download_button("Download Diversity Curve PNG", f, file_name="lexical_diversity_curve.png")

        st.subheader("Suggestions for Low-Diversity Zones")
        for zone in analysis_results["lexical_diversity"]["low_diversity_zones"]:
            st.text_area(f"Low Diversity Zone (UWR: {zone['uwr']:.3f})", value=zone["snippet"], height=100)
            st.write(f"**Top 5 common words:** {', '.join([f'{k} ({v})' for k, v in zone['top_common'].items()])}")
            st.write("_Consider replacing repeated words, adding a concrete sensory detail, or shortening multiple long sentences here to vary rhythm. Remember to preserve your authorial voice._")

        st.subheader("JSON Summary")
        st.json(analysis_results)

# --- Demo Mode ---
if __name__ == '__main__':
    print("Running in demo mode...")
    sample_text = "This is a sample text for testing purposes. It has several sentences of varying lengths. This sentence is a bit longer than the first one. The final sentence is short. This is a sample text for testing purposes. It has several sentences of varying lengths. This sentence is a bit longer than the first one. The final sentence is short."
    
    analysis_results = analyze_text(sample_text, 10, 5, 5, False, True)
    hist_path, lex_path = create_visuals(analysis_results)
    pdf_path = create_pdf_summary(analysis_results, hist_path, lex_path)
    json_path = create_json_summary(analysis_results, hist_path, lex_path, pdf_path)

    print(f"Generated files in: {DEFAULT_OUTPUT_DIR}")
    print(f"  - {hist_path}")
    print(f"  - {lex_path}")
    print(f"  - {pdf_path}")
    print(f"  - {json_path}")
