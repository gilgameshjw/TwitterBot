#!/bin/bash

# Read the YAML configuration file
config_file="config.yaml"
openai_engine=""
twitter_handle=""
num_tweets=""
character=""

while IFS=': ' read -r key value; do
if [[ $key == "openai_engine" ]]; then
    openai_engine="$value"
  elif [[ $key == "twitter_handle" ]]; then
    twitter_handle="$value"
  elif [[ $key == "num_tweets" ]]; then
    num_tweets="$value"
  elif [[ $key == "character" ]]; then
    character="$value"
  fi
done < "$config_file"


# Print the values
echo "openai_engine: $openai_engine"
echo "twitter_handle: $twitter_handle"
echo "num_tweets: $num_tweets"

# Check if twitter_api_key is "none" and update the value
echo "1. runs: python random_tweets.py" $character $twitter_handle
python random_tweets.py $character $twitter_handle 

echo "2. runs: python parse_tweets_data.py" $twitter_handle
python parse_tweets_data.py $twitter_handle

