# Essay Rhythm Analyzer

A Streamlit web application that measures the authenticity and engagement of writing by analyzing its linguistic rhythm, sentence flow, and lexical diversity.  
It does not detect AI. It helps writers understand how natural, varied, and readable their essays feel — the same qualities that make writing compelling to human readers.

Live app: [https://essayanalyzer.streamlit.app/](https://essayanalyzer.streamlit.app/)

---

## Overview

The analyzer evaluates essays using quantifiable stylistic metrics:
- Sentence-length rhythm – flow, pacing, and variation  
- Lexical diversity – freshness and range of word choice  
- Monotony detection – where writing rhythm or word variety collapses  
- Engagement visualization – clear plots and a one-page summary  

This is a writing improvement tool, not an AI classifier.  
It is designed for transparency, self-reflection, and stylistic refinement.

---

## Features

### Input
- Paste essay text or upload a `.txt` file directly on the app.  
- Adjustable analysis parameters:
  - Window size (words)
  - Stride (words)
  - Histogram bins  
- Optional **spaCy mode** for precise sentence splitting and tokenization.

### Analytics

**Sentence Length Distribution**  
- Calculates mean, standard deviation, coefficient of variation, and entropy.  
- Produces an amber-colored histogram visualizing rhythm variety.  

**Lexical Diversity Curve**  
- Measures unique-word-ratio (UWR) across sliding windows of text.  
- Highlights low-diversity “fatigue zones” in red.  

**Low-Diversity Zone Detection**  
- Pinpoints repetitive sections and generates actionable style suggestions:  
  “Consider varying sentence rhythm or replacing repeated words here.”

---

## Exports (Session-Based)

- `sentence_length_hist.png` — sentence-length histogram  
- `lexical_diversity_curve.png` — lexical diversity curve  
- `essay_onepager.pdf` — one-page visual + text summary  
- `analysis_summary.json` — structured numerical data + suggestions  

**Privacy note:**  
All analyses are session-based. Your text and results exist only temporarily in your browser session — nothing is stored, logged, or uploaded to the repository.

---

## Ethics and Privacy

The Essay Rhythm Analyzer prioritizes academic transparency and user privacy:
- Your essay is never stored or transmitted.
- No cloud database or tracking is used.
- The public GitHub repository contains only the open-source code.
- The hosted app ([https://essayanalyzer.streamlit.app/](https://essayanalyzer.streamlit.app/)) runs fully isolated per user session.

This project exists to make writing analysis transparent, not to support plagiarism detection or AI policing.
