import openai
import random
import json

def generate_synthetic_tweets(num_tweets):
    openai.api_key = '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'  # Replace with your OpenAI API key
    
    tweets = []
    
    for _ in range(num_tweets):
        prompt = "Generate a random tweet:"
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7
        )
        
        tweet_text = response.choices[0].text.strip()
        
        tweet_data = {
            "id": None,  # Replace with a unique ID for the tweet
            "id_str": None,  # Replace with a string representation of the tweet ID
            "user": {
                "id": None,  # Replace with a unique ID for the user
                "id_str": None,  # Replace with a string representation of the user ID
                "name": "Random User",
                "screen_name": "random_user",
                "location": None,
                "profile_location": None,
                "description": None,
                "url": None,
                "entities": {
                    "url": {
                        "urls": []
                    },
                    "description": {
                        "urls": []
                    }
                },
                "protected": False,
                "followers_count": None,  # Replace with the number of followers
                "friends_count": None,  # Replace with the number of friends
                "listed_count": None,  # Replace with the number of lists
                "created_at": None,  # Replace with the user's creation date
                "favourites_count": None,  # Replace with the number of favorites
                "utc_offset": None,
                "time_zone": None,
                "geo_enabled": None,
                "verified": None,
                "statuses_count": None,  # Replace with the number of statuses
                "lang": None,
                "contributors_enabled": None,
                "is_translator": None,
                "is_translation_enabled": None,
                "profile_background_color": None,
                "profile_background_image_url": None,
                "profile_background_image_url_https": None,
                "profile_background_tile": None,
                "profile_image_url": None,
                "profile_image_url_https": None,
                "profile_banner_url": None,
                "profile_link_color": None,
                "profile_sidebar_border_color": None,
                "profile_sidebar_fill_color": None,
                "profile_text_color": None,
                "profile_use_background_image": None,
                "has_extended_profile": None,
                "default_profile": None,
                "default_profile_image": None,
                "following": None,
                "follow_request_sent": None,
                "notifications": None,
                "translator_type": None
            },
            "text": tweet_text
        }
        
        tweets.append(tweet_data)
    
    return tweets

# Generate 100 random and different synthetic tweets
num_tweets = 5
synthetic_tweets = generate_synthetic_tweets(num_tweets)

# Print the generated tweets
for tweet in synthetic_tweets:
    print(json.dumps(tweet, indent=4))
    print("--------------------------------------")





