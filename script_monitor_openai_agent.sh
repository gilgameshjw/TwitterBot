#!/bin/bash

# Read the YAML configuration file
config_file="config.yaml"
openai_key=""
openai_engine=""
fine_tune_job_id=""

while IFS=': ' read -r key value; do
  if [[ $key == "openai_key" ]]; then
    openai_key="$value"
  elif [[ $key == "openai_engine" ]]; then
    openai_engine="$value"
  elif [[ $key == "fine_tune_job_id" ]]; then
    fine_tune_job_id="$value"
  fi
done < "$config_file"


# Print the values
echo "openai_key: $openai_key"
echo "openai_engine: $openai_engine"
echo "fine_tune_job_id: $fine_tune_job_id"

echo "1. set up the openai key"
export OPENAI_API_KEY=$openai_key

echo "2. runs: openai api fine_tunes.get -i "$fine_tune_job_id""
openai api fine_tunes.get -i $fine_tune_job_id
