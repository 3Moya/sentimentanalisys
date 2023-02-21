import os
import tweepy
from dotenv import load_dotenv

load_dotenv()

class TwitterClient:
    def __init__(self):
        self.client = tweepy.Client(bearer_token=os.getenv('BEARER_TOKEN'))

    def search_recent_tweets(self, limit, query):
        return tweepy.Paginator(
            self.client.search_recent_tweets,
            query=query,
            tweet_fields=['created_at', 'lang', 'public_metrics'],
            max_results=100
        ).flatten(limit=limit)
