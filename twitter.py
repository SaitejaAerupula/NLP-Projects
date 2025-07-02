import tweepy
from textblob import TextBlob

# === Step 1: Twitter Developer Access ===
# Replace with your actual Twitter Developer API keys
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# === Step 2: Set Up Tweepy API ===
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# === Step 3: Search for Tweets ===
query = 'Python programming'
num_tweets = 100

tweets = api.search_tweets(q=query, lang='en', count=num_tweets)

# === Step 4: Sentiment Analysis ===
for tweet in tweets:
    text = tweet.text
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity

    if sentiment > 0:
        sentiment_type = 'Positive 😊'
    elif sentiment < 0:
        sentiment_type = 'Negative 😠'
    else:
        sentiment_type = 'Neutral 😐'

    print(f"Tweet: {text}\nSentiment: {sentiment_type}\n")
