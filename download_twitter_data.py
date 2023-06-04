
import sys
import json
import tweepy
import yaml


# get the above credentials from config.yaml file
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# get the handle name from the command line
if len(sys.argv) != 2:
    print("Usage: python download_twitter_data.py <twitter_handle>")
    sys.exit(1)
twitter_handle = sys.argv[1]

# Set the Twitter handle for which you want to download data
twitter_handle = 'twitter_handle'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create the API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Get all tweets for the specified user
tweets = []
for tweet in tweepy.Cursor(api.user_timeline, screen_name=twitter_handle, tweet_mode='extended').items():
    tweets.append(tweet._json)

# Save the tweets to a file (JSONL format)
filename = 'data/twitter_data_'+twitter_handle+'.jsonl'
with open(filename, 'w') as file:
    for tweet in tweets:
        file.write(json.dumps(tweet)+'\n')

print(f'Twitter data downloaded and saved to {filename}')

