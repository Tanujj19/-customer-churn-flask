import os
import pickle
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, render_template
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
MODEL_CACHE = "flask_model.pkl"

# ── Build / load model ────────────────────────────────────────────────────────
def get_model():
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)

    # Load sample data for training
    df = pd.read_csv("sample_reviews.csv")
    df.dropna(subset=['UserId', 'ProductId', 'Text', 'Score', 'sentiment'], inplace=True)

    # Customer features
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

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error  = None

    if request.method == "POST":
        try:
            avg_rating     = float(request.form["avg_rating"])
            purchase_count = float(request.form["purchase_count"])
            avg_sentiment  = float(request.form["avg_sentiment"])

            if not (1 <= avg_rating <= 5):
                raise ValueError("Avg Rating must be between 1 and 5.")
            if purchase_count < 0:
                raise ValueError("Purchase Count cannot be negative.")
            if not (-1 <= avg_sentiment <= 1):
                raise ValueError("Avg Sentiment must be between -1 and 1.")

            features = pd.DataFrame([[avg_rating, purchase_count, avg_sentiment]],
                                     columns=['avg_rating', 'purchase_count', 'avg_sentiment'])
            pred  = model.predict(features)[0]
            prob  = model.predict_proba(features)[0][1]
            result = {
                "prediction": "⚠️ High Risk of Churn" if pred == 1 else "✅ Low Risk of Churn",
                "probability": f"{prob:.2%}",
                "risk": pred
            }
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)