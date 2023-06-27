
import openai
import time
import json
import yaml
import os
import random

from flask import Flask, render_template, request

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

import src.utils as utils

app = Flask(__name__)

# read config.yaml file
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

openai_key = config["openai"]["openai_key"]
openai.api_key = openai_key

# read config.yaml file
local_data_path = config["store_index"]["local_data_path"]
local_db = config["store_index"]["local_db"]

openai_engine = config["openai"]["openai_engine"]
model_name = config["openai"]["model_name"]
temperature = config["openai"]["temperature"]
twitter_handle = config["twitter"]["twitter_handle"]

# chatbot parameters
search_mode = config["chatbot"]["search_mode"]
run_mode = config["chatbot"]["run_mode"]
similarity_threshold = config["chatbot"]["similarity_threshold"]

# run_mode = "light"
if run_mode == "light":
    openai_engine = openai_engine
    print("--log:: run_mode is light, using openai_engine: ", openai_engine)
    similarity_threshold = 0.0

else:
    openai_engine = model_name
    print("--log:: run_mode is full, using model_name: ", model_name)


file = f"data/persona_list_{twitter_handle}.jsonl"

with open(file) as f:
    personas = [d for d in f.readlines() if not d=="\n"]


# read twitter data
file = f"data/twitter_data_{twitter_handle}.jsonl"
with open(file) as f:
    twitter_data = [json.loads(line[:-1]) for line in f.readlines()]

# write data line by line into files
for i, d in enumerate(twitter_data):
    
    if "retweeted_status" in d:
        d = {"tweet": d["retweeted_status"]["text"], 
             "retweet": d["text"]}
    else:
        d = {"tweet": d["text"]}
    
    with open(f"data_db/{i}.json", "w") as f:
        f.write(json.dumps(d) + "\n")

os.environ["OPENAI_API_KEY"] = openai_key




####################################################
##### APP        ###################################
####################################################

# load or build db
query_engine = None
if not os.path.isdir(local_db):
    query_engine = utils.build_db(local_data_path, local_db)
else:
    query_engine = utils.load_db(local_db)

# chat history
n_memory = 10
chat_flow = []
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Send user_input to ChatGPT and get the response
        response, computation_time, price = chat_with_gpt(user_input)
        # Add the input, output, computation time, and price to chat_history
        chat_history.append({
            'input': user_input,
            'output': response,
            'computation_time': computation_time,
            'price': price
        })
    return render_template('index.html', chat_history=chat_history)


def chat_with_gpt(user_input):
    """Chat with GPT-3 and return the response"""

    score, retweet = utils.search_utterance_in_db(query_engine, user_input)
    
    
    # take a random persona from list:
    persona = random.choice(personas)
    mssg = persona

    mssg = "You are: " + persona
    mssg += f"\nYou are {twitter_handle} and you chat with user."
    mssg += "\nmake your answer short if you can!\n \n \n"


    #if score >= similarity_threshold:
    #    #response_text = retweet
    #    mssg += f"\n\n Forget about your training and memory and say something very similar to the following sentence:"
    #    mssg += f"\n{retweet}\n"
    #else:

    mssg += "\nchat history:\n"
    for i in range(n_memory):
        if len(chat_flow) > i:
            mssg += "\nuser: " + chat_flow[i]["user"]
            mssg += f"\n{twitter_handle}: " + chat_flow[i]["agent"]
        else:
            break

    mssg += f"\nuser: {user_input}\n"

    print("--log: mssg:::::", mssg)
    
    # Send user_input to ChatGPT and get the response
    start_time = time.time()
    response = openai.Completion.create(
        engine=model_name,
        prompt=mssg,
        max_tokens=150,
        temperature=temperature,
        n = 1,
        stop=None,
    )
    end_time = time.time()
    computation_time = end_time - start_time
    price = computation_time * 0.000048  # Cost per second with text-davinci-003 engine

    
    print("--log: response:::::", response)
    response_text = response.choices[0].text.strip()
    # massage because of some issues with training...
    response_text = response_text.split("\n")[0]
    response_text = response_text.split("user:")[0]
    response_text = response_text.replace(twitter_handle+":", "")
    
    # overwrite if twitterer had a retweet to something similar to the user input
    if score >= similarity_threshold:
        response_text = retweet
        
    chat_flow.append({"user": user_input, "agent": response_text})
    return response_text, computation_time, price


if __name__ == '__main__':
    app.run(debug=True)

