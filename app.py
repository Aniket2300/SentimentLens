from flask import Flask, render_template, request, jsonify, send_file
import ollama
import json
import re
import io
import csv as csv_module
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

app = Flask(__name__)

# Keep the last analyzed reviews in memory so /wordcloud and /export-report can reuse them
last_analysis = {'reviews': [], 'result': None}


@app.route('/')
def home():
    return render_template('index.html')


def run_sentiment_analysis(reviews):
    """Shared analysis logic used by both /analyze and /analyze-csv."""
    if len(reviews) == 0:
        return {'error': 'No valid reviews found'}

    numbered_reviews = "\n".join([f"{i+1}. {r}" for i, r in enumerate(reviews)])

    prompt = f"""You are a sentiment analysis assistant. Analyze the following reviews.

Reviews:
{numbered_reviews}

For EACH review, classify it as exactly one of: "Positive", "Negative", "Neutral".

Then separately:
- Look ONLY at the Positive reviews and summarize what they praise, in one short sentence. If there are no Positive reviews, write "None noted".
- Look ONLY at the Negative reviews and summarize what they complain about, in one short sentence. If there are no Negative reviews, write "None noted".

These two summaries must be independent of each other.

Return ONLY valid JSON, no other text, no markdown formatting, in this exact format:
{{
    "classifications": [
        {{"review_number": 1, "sentiment": "Positive"}},
        {{"review_number": 2, "sentiment": "Negative"}}
    ],
    "common_praise": "<one sentence>",
    "common_complaints": "<one sentence>"
}}

Make sure "classifications" has exactly {len(reviews)} entries, one per review, in order.
"""

    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        raw = response['message']['content'].strip()

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return {'error': 'Model did not return valid JSON'}

        json_str = re.sub(r',\s*([}\]])', r'\1', match.group(0))
        result = json.loads(json_str)

        classifications = result.get('classifications', [])
        for c in classifications:
            idx = c.get('review_number', 0) - 1
            if 0 <= idx < len(reviews):
                c['text'] = reviews[idx]

        total = len(classifications)
        counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for c in classifications:
            sentiment = c.get('sentiment', 'Neutral')
            if sentiment in counts:
                counts[sentiment] += 1

        breakdown = {
            k: round((v / total) * 100, 1) if total > 0 else 0
            for k, v in counts.items()
        }

        common_praise = result.get('common_praise', '')
        common_complaints = result.get('common_complaints', '')

        # Fallback: if the model said "None noted" but matching reviews actually exist,
        # ask again using only those specific reviews to force a real summary.
        positive_texts = [c['text'] for c in classifications if c.get('sentiment') == 'Positive']
        negative_texts = [c['text'] for c in classifications if c.get('sentiment') == 'Negative']

        if 'none noted' in common_praise.lower() and positive_texts:
            common_praise = summarize_subset(positive_texts, "praise")

        if 'none noted' in common_complaints.lower() and negative_texts:
            common_complaints = summarize_subset(negative_texts, "complaints")

        return {
            'classifications': classifications,
            'breakdown': breakdown,
            'common_praise': common_praise,
            'common_complaints': common_complaints,
            'total_reviews': total
        }
    except Exception as e:
        return {'error': f'Failed to analyze reviews: {str(e)}'}


def summarize_subset(texts, kind):
    """Fallback summarizer: re-asks Ollama using only the given reviews to force a real answer."""
    joined = "\n".join(f"- {t}" for t in texts)
    prompt = f"""These are reviews that all express {kind}:
{joined}

Write ONE short sentence summarizing the common {kind} theme across them. Respond with only the sentence, no preamble.
"""
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip().strip('"')
    except Exception:
        return texts[0]  # last resort: just show the first review verbatim


@app.route('/analyze', methods=['POST'])
def analyze():
    reviews_text = request.json.get('reviews', '').strip()
    if not reviews_text:
        return jsonify({'error': 'No reviews provided'}), 400

    reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
    result = run_sentiment_analysis(reviews)

    if 'error' in result:
        return jsonify(result), 500

    last_analysis['reviews'] = reviews
    last_analysis['result'] = result
    return jsonify(result)


@app.route('/analyze-csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        stream = io.StringIO(file.stream.read().decode('utf-8'))
        reader = csv_module.DictReader(stream)

        # Try to find a column that looks like it holds review text
        candidate_columns = ['review', 'text', 'comment', 'feedback', 'description']
        fieldnames = [f.lower() for f in (reader.fieldnames or [])]
        target_col = None
        for candidate in candidate_columns:
            if candidate in fieldnames:
                target_col = reader.fieldnames[fieldnames.index(candidate)]
                break

        if not target_col and reader.fieldnames:
            target_col = reader.fieldnames[0]  # fallback to first column

        reviews = []
        stream.seek(0)
        reader = csv_module.DictReader(stream)
        for row in reader:
            val = row.get(target_col, '').strip()
            if val:
                reviews.append(val)

        if len(reviews) == 0:
            return jsonify({'error': 'No reviews found in CSV'}), 400

        # Cap at 50 reviews to keep the LLM prompt reasonable
        reviews = reviews[:50]

        result = run_sentiment_analysis(reviews)
        if 'error' in result:
            return jsonify(result), 500

        last_analysis['reviews'] = reviews
        last_analysis['result'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to process CSV: {str(e)}'}), 500


@app.route('/wordcloud', methods=['GET'])
def wordcloud():
    reviews = last_analysis.get('reviews', [])
    if not reviews:
        return jsonify({'error': 'No analysis available yet'}), 400

    try:
        text = " ".join(reviews)
        wc = WordCloud(width=800, height=300, background_color='#0d1520',
                        colormap='cool', max_words=60).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Failed to generate word cloud: {str(e)}'}), 500


@app.route('/compare', methods=['POST'])
def compare():
    batch_a_text = request.json.get('batch_a', '').strip()
    batch_b_text = request.json.get('batch_b', '').strip()

    if not batch_a_text or not batch_b_text:
        return jsonify({'error': 'Both batches are required'}), 400

    reviews_a = [r.strip() for r in batch_a_text.split('\n') if r.strip()]
    reviews_b = [r.strip() for r in batch_b_text.split('\n') if r.strip()]

    result_a = run_sentiment_analysis(reviews_a)
    result_b = run_sentiment_analysis(reviews_b)

    if 'error' in result_a or 'error' in result_b:
        return jsonify({'error': 'Failed to analyze one or both batches'}), 500

    return jsonify({'batch_a': result_a, 'batch_b': result_b})


@app.route('/export-report', methods=['GET'])
def export_report():
    result = last_analysis.get('result')
    if not result:
        return jsonify({'error': 'No analysis available yet'}), 400

    try:
        width, height = 800, 500
        card = Image.new('RGB', (width, height), color=(10, 15, 20))
        draw = ImageDraw.Draw(card)

        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            label_font = ImageFont.truetype("arial.ttf", 16)
            text_font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            title_font = label_font = text_font = ImageFont.load_default()

        draw.text((30, 30), "SentimentLens Report", font=title_font, fill=(56, 189, 248))

        breakdown = result['breakdown']
        y = 100
        draw.text((30, y), f"Total reviews analyzed: {result['total_reviews']}", font=text_font, fill=(230, 242, 247))
        y += 40
        draw.text((30, y), f"Positive: {breakdown['Positive']}%   Negative: {breakdown['Negative']}%   Neutral: {breakdown['Neutral']}%",
                   font=text_font, fill=(230, 242, 247))

        y += 60
        draw.text((30, y), "COMMON PRAISE", font=label_font, fill=(74, 222, 128))
        y += 26
        draw.text((30, y), result['common_praise'], font=text_font, fill=(230, 242, 247))

        y += 60
        draw.text((30, y), "COMMON COMPLAINTS", font=label_font, fill=(248, 113, 113))
        y += 26
        draw.text((30, y), result['common_complaints'], font=text_font, fill=(230, 242, 247))

        img_io = io.BytesIO()
        card.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='sentiment_report.png')
    except Exception as e:
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5003)