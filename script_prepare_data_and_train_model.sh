#!/bin/bash

# Read the YAML configuration file
config_file="config.yaml"
openai_key=""
openai_train_engine=""
twitter_handle=""

while IFS=': ' read -r key value; do
  if [[ $key == "openai_key" ]]; then
    openai_key="$value"
  elif [[ $key == "openai_train_engine" ]]; then
    openai_train_engine="$value"
  elif [[ $key == "twitter_handle" ]]; then
    twitter_handle="$value"
  fi
done < "$config_file"


# Print the values
echo "openai_key: $openai_key"
echo "twitter_handle: $twitter_handle"
echo "openai_engine: $openai_train_engine"


echo "1. set up the openai key"
export OPENAI_API_KEY=$openai_key

echo "2. runs: openai tools fine_tunes.prepare_data -f data/parsed_twitter_data_"$twitter_handle".jsonl"
openai tools fine_tunes.prepare_data -f data/parsed_twitter_data_"$twitter_handle".jsonl

echo "3. runs: openai api fine_tunes.create -t data/parsed_twitter_data_"$twitter_handle"_prepared.jsonl -m "$openai_train_engine""
openai api fine_tunes.create -t data/parsed_twitter_data_"$twitter_handle"_prepared.jsonl -m $openai_train_engine
