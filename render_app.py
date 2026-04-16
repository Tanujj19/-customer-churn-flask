import os
import pickle
import pandas as pd
import nltk
from flask import Flask, request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)

def get_model():
    MODEL_CACHE = "flask_model.pkl"
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv("sample_reviews.csv")
    df.dropna(subset=['UserId', 'ProductId', 'Score', 'sentiment'], inplace=True)

    customer_df = df.groupby('UserId').agg(
        avg_rating=('Score', 'mean'),
        purchase_count=('ProductId', 'count'),
        avg_sentiment=('sentiment', 'mean')
    ).reset_index()

    customer_df['churn'] = (
        (customer_df['purchase_count'] < 2) & (customer_df['avg_sentiment'] < 0)
    ).astype(int)

    FEATURES = ['avg_rating', 'purchase_count', 'avg_sentiment']
    X = customer_df[FEATURES]
    y = customer_df['churn']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline

model = get_model()

def render_page(avg_rating="", purchase_count="", avg_sentiment="", result_html="", error_html=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Customer Churn Predictor</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
                display: flex; justify-content: center; align-items: center;
                min-height: 100vh; padding: 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 40px;
                 width: 100%; max-width: 480px;
                 box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        h1 {{ font-size: 1.5rem; margin-bottom: 6px; color: #1a1a2e; }}
        p.subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 28px; }}
        label {{ display: block; font-size: 0.85rem; font-weight: 600;
                 color: #333; margin-bottom: 4px; }}
        input[type="number"] {{ width: 100%; padding: 10px 14px;
                                border: 1px solid #ddd; border-radius: 8px;
                                font-size: 0.95rem; margin-bottom: 6px; }}
        small {{ color: #999; font-size: 0.78rem; display: block; margin-bottom: 14px; }}
        button {{ width: 100%; padding: 12px; background: #4f8ef7; color: white;
                  border: none; border-radius: 8px; font-size: 1rem;
                  font-weight: 600; cursor: pointer; }}
        button:hover {{ background: #3a7de0; }}
        .result {{ margin-top: 24px; padding: 16px 20px; border-radius: 8px;
                   font-size: 1rem; font-weight: 600; }}
        .high {{ background: #fff3cd; border-left: 4px solid #f0ad4e; color: #856404; }}
        .low  {{ background: #d4edda; border-left: 4px solid #28a745; color: #155724; }}
        .prob {{ font-size: 0.85rem; font-weight: 400; margin-top: 4px; }}
        .error {{ margin-top: 16px; padding: 12px 16px; background: #f8d7da;
                  border-left: 4px solid #dc3545; border-radius: 8px;
                  color: #721c24; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🧠 Customer Churn Predictor</h1>
        <p class="subtitle">Enter customer features to predict churn risk.</p>
        <form method="POST">
            <label>Average Rating</label>
            <input type="number" name="avg_rating" step="0.01" min="1" max="5"
                   placeholder="e.g. 4.2" value="{avg_rating}" required/>
            <small>Range: 1.0 – 5.0</small>
            <label>Purchase Count</label>
            <input type="number" name="purchase_count" step="1" min="0"
                   placeholder="e.g. 3" value="{purchase_count}" required/>
            <small>Total number of reviews/purchases</small>
            <label>Average Sentiment Score</label>
            <input type="number" name="avg_sentiment" step="0.001" min="-1" max="1"
                   placeholder="e.g. 0.35" value="{avg_sentiment}" required/>
            <small>Range: -1.0 (negative) to 1.0 (positive)</small>
            <button type="submit">Predict Churn</button>
        </form>
        {error_html}
        {result_html}
    </div>
</body>
</html>"""

@app.route("/", methods=["GET", "POST"])
def index():
    result_html = ""
    error_html  = ""
    avg_rating = purchase_count = avg_sentiment = ""

    if request.method == "POST":
        avg_rating     = request.form.get("avg_rating", "")
        purchase_count = request.form.get("purchase_count", "")
        avg_sentiment  = request.form.get("avg_sentiment", "")
        try:
            r = float(avg_rating)
            p = float(purchase_count)
            s = float(avg_sentiment)

            if not (1 <= r <= 5):
                raise ValueError("Avg Rating must be between 1 and 5.")
            if p < 0:
                raise ValueError("Purchase Count cannot be negative.")
            if not (-1 <= s <= 1):
                raise ValueError("Avg Sentiment must be between -1 and 1.")

            features = pd.DataFrame([[r, p, s]],
                                     columns=['avg_rating', 'purchase_count', 'avg_sentiment'])
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            css  = "high" if pred == 1 else "low"
            prediction = "⚠️ High Risk of Churn" if pred == 1 else "✅ Low Risk of Churn"
            result_html = f'<div class="result {css}">{prediction}<div class="prob">Churn Probability: {prob:.2%}</div></div>'
        except ValueError as e:
            error_html = f'<div class="error">❌ {str(e)}</div>'
        except Exception as e:
            error_html = f'<div class="error">❌ Unexpected error: {str(e)}</div>'

    return render_page(avg_rating, purchase_count, avg_sentiment, result_html, error_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
