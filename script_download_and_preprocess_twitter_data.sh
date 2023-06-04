#!/bin/bash

# Read the YAML configuration file
config_file="config.yaml"
twitter_handle=""

while IFS=': ' read -r key value; do
if [[ $key == "twitter_handle" ]]; then
    twitter_handle="$value"
  fi
done < "$config_file"

# Print the values
echo "twitter_handle: $twitter_handle"

echo "1. runs: download_twitter_data.py" $twitter_handle
python download_twitter_data.py $twitter_handle 

echo "2. runs: python parse_tweets_data_twitter.py" $twitter_handle
python parse_tweets_data_twitter.py $twitter_handle

