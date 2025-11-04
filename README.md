# 🧠 Essay Rhythm Analyzer

A **Streamlit web application** that quantifies the *human authenticity and engagement* of essays through linguistic rhythm and lexical variety.  
It’s designed for students, educators, and researchers who want to measure — not “detect” — how *alive, varied, and readable* writing feels.

---

## 🧩 Overview

The analyzer evaluates essays based on:
- **Sentence-length rhythm** (variation and flow)
- **Lexical diversity** (freshness and word variety)
- **Monotony detection** (where writing feels repetitive)
- **Engagement visualization** (plots + summaries)

Unlike AI detectors, this tool doesn’t label writing as “human” or “AI.”  
Instead, it helps writers **improve rhythm, variety, and clarity** while preserving their authentic voice.

---

## 🚀 Features

### ✍️ Input
- Paste your essay or upload a `.txt` file.
- Adjustable parameters for analysis:
  - Window size (words)
  - Stride (words)
  - Histogram bins
- Optional **spaCy mode** for advanced NLP sentence splitting and tokenization.

### 📊 Analytics
- **Sentence Length Distribution**
  - Computes mean, standard deviation, coefficient of variation, and entropy.
  - Amber histogram visualizes rhythm variety.

- **Lexical Diversity Curve**
  - Computes sliding-window *unique word ratios*.
  - Highlights low-diversity (“fatigue”) segments in red.

- **Suggestions**
  - Detects repetitive segments and generates localized feedback:
    > “Consider adding a sensory detail or varying sentence rhythm here.”

### 📄 Exports (Session-Scoped)
- `sentence_length_hist.png`
- `lexical_diversity_curve.png`
- `essay_onepager.pdf`
- `analysis_summary.json`

> ⚠️ **Important:**  
> No analysis files or user essays are stored permanently or uploaded anywhere.  
> All data is processed **in-memory** during your session and cleared when you close the browser.

---

## 🧠 Academic Integrity & Privacy

This project is designed for **self-analysis**, not content policing.  
No essay text or metadata is saved, logged, or shared.  
Your input remains private and ephemeral — nothing is committed to the public repository.

**Transparency:**  
- The public GitHub repository only contains source code.  
- The live Streamlit deployment runs isolated session storage.  
- Uploaded essays and generated files exist only temporarily per user session.

---

## 🧩 Repository Structure
