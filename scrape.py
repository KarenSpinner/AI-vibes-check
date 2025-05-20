import praw
import pandas as pd
from datetime import datetime, timedelta, UTC

# ✅ Your Reddit credentials
reddit = praw.Reddit(
    client_id="YOUR_ID_HERE",
    client_secret="YOUR_SECRET_HERE",
    user_agent="YOUR_AGENT_NAME_HERE"
)

# 🔍 Search setup
subreddits = [
    "marketing", "copywriting", "advertising", "FreelanceWriters",
    "Entrepreneur", "technology", "artificial", "digital_marketing",
    "smallbusiness", "dataisbeautiful", "productivity", "UXDesign",
    "ChatGPT", "OpenAI", "ClaudeAI", "QualityAssurance", "worldbuilding"
]

query = (
    '"artificial intelligence" OR ChatGPT OR GPT-4 OR Gemini OR Claude '
    'OR Midjourney OR DALL-E OR "generative AI" OR "AI tools"'
)

limit_per_subreddit = 200  # Adjust as needed

# Time filter: last 6 months
after_timestamp = int((datetime.now(UTC) - timedelta(days=180)).timestamp())

# Scrape posts
posts = []

for subreddit_name in subreddits:
    print(f"\n🔍 Searching r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)

    for post in subreddit.search(query, sort="new", limit=limit_per_subreddit, time_filter="all"):
        if post.created_utc >= after_timestamp:
            posts.append({
                "id": post.id,
                "subreddit": subreddit_name,
                "title": post.title,
                "text": post.selftext,
                "created": datetime.fromtimestamp(post.created_utc),
                "score": post.score,
                "num_comments": post.num_comments,
                "author": str(post.author),
                "url": post.url
            })

print(f"\n✅ Collected {len(posts)} posts.")
df = pd.DataFrame(posts)
df.to_csv("praw_ai_sentiment_posts.csv", index=False)
print("📁 Saved as praw_ai_sentiment_posts.csv in this folder.")

