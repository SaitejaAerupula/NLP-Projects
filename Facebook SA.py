import tweepy
from textblob import TextBlob

# Replace with your Twitter API credentials
CONSUMER_KEY = 'YOUR_CONSUMER_KEY'
CONSUMER_SECRET = 'YOUR_CONSUMER_SECRET'
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
ACCESS_TOKEN_SECRET = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Search for recent tweets containing a keyword
keyword = "python"
tweets = api.search_tweets(q=keyword, lang="en", count=100)

# Analyze sentiment for each tweet
for tweet in tweets:
    analysis = TextBlob(tweet.text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        label = "Positive"
    elif sentiment < 0:
        label = "Negative"
    else:
        label = "Neutral"
    print(f"Tweet: {tweet.text}")
    print(f"Sentiment: {label} (Score: {sentiment:.2f})")
    print("-" * 40)
