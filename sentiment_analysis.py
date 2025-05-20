import pandas as pd
import re
from textblob import TextBlob

# 🔁 Load your CSV
df = pd.read_csv("praw_ai_sentiment_posts.csv")

# 🧠 Combine title and text
df['full_text'] = df['title'].fillna('') + '. ' + df['text'].fillna('')

# 📅 Convert created column to datetime (optional but recommended)
df['created'] = pd.to_datetime(df['created'], errors='coerce')

# 💬 Sentiment Analysis using TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['full_text'].apply(get_sentiment)

# ❤️ Emotion Keyword Tagging
emotion_keywords = {
    'curiosity': ['curious', 'wonder', 'try', 'testing', 'experiment', 'exploring', 'idea', 'learn'],
    'excitement': ['excited', 'amazing', 'cool', 'love', 'fantastic', 'awesome'],
    'hope': ['hope', 'maybe', 'potential', 'possibility', 'dream'],
    'concern': ['problem', 'concern', 'worry', 'issue', 'risk', 'danger'],
    'skepticism': ['doubt', 'not sure', 'don’t know', 'unclear', 'unconvinced'],
    'frustration': ['annoyed', 'frustrated', 'hate', 'angry', 'irritating', 'broken'],
    'fear': ['scared', 'terrified', 'fear', 'afraid']
}

def tag_emotions(text):
    detected = set()
    lowered = text.lower()
    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if re.search(r'\b' + re.escape(word) + r'\b', lowered):
                detected.add(emotion)
    return ', '.join(sorted(detected)) if detected else 'neutral'

df['emotions'] = df['full_text'].apply(tag_emotions)

# 📊 Summary stats
sentiment_counts = df['sentiment'].value_counts()
emotion_counts = df[df['emotions'] != 'neutral']['emotions'].str.split(', ').explode().value_counts()

# 🔢 Display results
print("\n📈 Sentiment Distribution:")
print(sentiment_counts)
print("\n📈 Emotion Distribution (non-neutral):")
print(emotion_counts)

# 💾 Save results if needed
df.to_csv("ai_sentiment_emotion_labeled.csv", index=False)
