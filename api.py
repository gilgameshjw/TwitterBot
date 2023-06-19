
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
deeplake_key = config["deeplake"]["api_key"]
deeplake_account_name = config["deeplake"]["account_name"]
deeplake_local_path = config["deeplake"]["local_path"]
deeplake_search_nk = config["deeplake"]["search_nk"]
deeplake_search_model = config["deeplake"]["search_model"]

openai_engine = config["openai"]["openai_engine"]
model_name = config["openai"]["model_name"]
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
file = f"data/train_twitter_data_{twitter_handle}.jsonl"
with open(file) as f:
    twitter_data = [json.loads(line[:-1]) for line in f.readlines()]

# write data line by line into files
for i, d in enumerate(twitter_data):
    with open(f"data_db/{i}.json", "w") as f:
        f.write(json.dumps(d) + "\n")

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["ACTIVELOOP_TOKEN"] = deeplake_key


####################################################
##### APP        ###################################
####################################################

# search model
search_model = ChatOpenAI(model_name=deeplake_search_model) 
# build qa
print(deeplake_local_path)
qa = utils.get_qa(search_model, deeplake_local_path, deeplake_account_name)

# chat history
n_memory = 10
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

    """
    # search in memory for similar prompts
    def search_in_memory(m_prompts, utterance):
        v_utterance = np.array(embeddings.embed_query(utterance))
        v_scores = m_prompts.dot(v_utterance)
        id, score = sorted(enumerate(v_scores.tolist()), key=lambda x: x[1], reverse=True)[0]
        return id, score
    
    id, score = search_in_memory(m_prompts, user_input)
    print("--log:: retrieven id: ", id, "score: ", score)
    
    """
    
    qa = utils.get_qa(search_model, deeplake_local_path, deeplake_account_name)
    
    # take a random persona from list:
    persona = random.choice(personas)
    mssg = persona
    # add history to mssg
    for i in range(n_memory):
        if len(chat_history) > i:
            mssg += "\n" + chat_history[-i-1]["user"]
            mssg += "\n" + chat_history[-i-1]["agent"]
    # hash tag analyser
    mssg += utils.hash_tag_analyser(search_model, qa, user_input)
    # date analyser
    mssg += utils.date_analyser(search_model, qa, user_input)
    # content analyser
    mssg += utils.content_analyser(search_model, qa, user_input)

    # Send user_input to ChatGPT and get the response
    start_time = time.time()
    response = openai.Completion.create(
        engine=openai_engine,
        prompt=mssg,
        max_tokens=50,
        temperature=0.7,
        n = 1,
        stop=None,
    )
    end_time = time.time()
    computation_time = end_time - start_time
    price = computation_time * 0.000048  # Cost per second with text-davinci-003 engine

    response_text = response.choices[0].text.strip()
    chat_history.append({"user": user_input, "agent": response_text})
    print(chat_history)
    return response_text, computation_time, price


if __name__ == '__main__':
    app.run(debug=True)

