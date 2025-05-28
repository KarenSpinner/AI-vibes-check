import pandas as pd
from transformers import pipeline

# Load your source file
df = pd.read_csv("praw_ai_sentiment_posts.csv")

# Prepare text
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['content'] = df['title'] + " " + df['text']
texts = df['content'].astype(str).tolist()

# Load sentiment classifier (POS/NEU/NEG)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Load emotion classifier (28 emotions)
emotion_pipeline = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    return_all_scores=True
)

# Setup batching
batch_size = 100
sentiment_labels, sentiment_scores = [], []
emotion_top1, emotion_top2 = [], []

def extract_top_emotions(scores, top_n=2):
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    return [e['label'] for e in sorted_scores[:top_n]]

# Process in batches
for i in range(0, len(texts), batch_size):
    print(f"🔄 Batch {i}–{i+batch_size}")
    batch = texts[i:i+batch_size]

    # Sentiment
    s_results = sentiment_pipeline(batch, truncation=True, max_length=512)
    sentiment_labels += [r['label'] for r in s_results]
    sentiment_scores += [r['score'] for r in s_results]

    # Emotion
    e_results = emotion_pipeline(batch, truncation=True, max_length=512)
    emotion_top1 += [extract_top_emotions(r)[0] for r in e_results]
    emotion_top2 += [extract_top_emotions(r)[1] for r in e_results]

# Add to DataFrame
df['sentiment_label'] = sentiment_labels
df['sentiment_score'] = sentiment_scores
df['emotion_1'] = emotion_top1
df['emotion_2'] = emotion_top2

# Save updated file
df.to_csv("praw_ai_posts_sentiment_and_emotions_NEW.csv", index=False)
print("✅ Done! Output saved.")
