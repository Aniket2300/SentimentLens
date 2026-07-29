# SentimentLens

A local AI-powered sentiment analysis tool for reviews. Paste in reviews (or upload a CSV), and get an instant sentiment breakdown, common praise/complaints, a word cloud, and side-by-side comparisons between two review batches — all running locally, no cloud APIs required.

## How it works

1. Reviews are sent to a local LLM (Llama 3.2 via Ollama) which classifies each one as Positive, Negative, or Neutral.
2. The model also summarizes common praise (from Positive reviews) and common complaints (from Negative reviews) independently, so one doesn't bleed into the other.
3. A fallback mechanism re-queries the model with only the relevant subset of reviews if it initially fails to produce a summary, improving reliability with a small local model.
4. Percentage breakdowns are computed in Python from the classifications, not trusted from the LLM's own math.

## Features

- Paste reviews (one per line) or upload a CSV — auto-detects the review text column
- Sentiment breakdown shown as an interactive doughnut chart
- Click any chart segment to filter the review list to just that sentiment
- Word cloud of the most frequent terms across all reviews
- Compare two batches of reviews side by side (e.g. before/after a product update, or two competing products)
- Export a downloadable PNG report card summarizing the analysis
- Dark, animated UI with toast notifications, drag-and-drop CSV upload, and smooth transitions

## Tech stack

- **Backend**: Flask
- **Language model**: Llama 3.2 via [Ollama](https://ollama.com) (runs locally)
- **Word cloud**: `wordcloud` Python library
- **Image generation**: Pillow (for the exportable report card)
- **Frontend**: HTML/CSS/JS, Chart.js for visualizations, canvas-based particle background

## Setup

1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull llama3.2
   ```

2. Clone this repo and install dependencies:
   ```bash
   git clone https://github.com/Aniket2300/SentimentLens.git
   cd SentimentLens
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

4. Start the app:
   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5003` and paste in some reviews to get started.

## CSV format

Any CSV with a column named `review`, `text`, `comment`, `feedback`, or `description` will work. If none of those are found, the first column is used.

## Author

Built by [Aniket Jaiswal](https://github.com/Aniket2300) as part of an ongoing AI/ML portfolio project series.