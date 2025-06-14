# AI-vibes-check
This project involved scraping Reddit posts that discussed AI and related topics for sentiment analysis. 
All raw data was deleted immediately after analysis in keeping with Reddit's TOS.
You may not sell raw data scraped from Reddit.

See Reddit's Data API Wiki for more details on how to access and handle Reddit data via API: 
https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki

This repository includes:
1. Python script for scraping posts via Reddit's API. You will need API credentials to run it. (**scrape.py**)
2. Python script for sentiment analysis using TextBlob. (**sentiment_analysis.py**)
3. Python script combining (1) sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment and (2) emotional classification using SamLowe/roberta-base-go_emotions (**advanced_sentiment_and_emotions.py**) 

All the scripts were written with help from ChatGPT and tested in my local Mac environment.

## How to use
If you're totally new to Python and/or Reddit's API, I suggest pasting the scripts into ChatGPT 4o or Claude 3.7 Sonnet and asking them to walk you through accessing the command line on your local machine and installing Python plus the necessary libraries. 



