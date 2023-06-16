
import openai
import time
import json
import yaml
import os
import tqdm 
import random

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 
from flask import Flask, render_template, request
import numpy as np
import pickle as pkl


app = Flask(__name__)

# read config.yaml file
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

api_key = config["openai"]["openai_key"]
openai.api_key = api_key
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


######################################################
######################################################
file = "data/persona_list_realDonaldTrump.jsonl"

with open(file) as f:
    personas = [d for d in f.readlines() if not d=="\n"]


# read twitter data
file = "data/train_twitter_data_realDonaldTrump.jsonl"
with open(file) as f:
    twitter_data = [json.loads(line[:-1]) for line in f.readlines()]

# write data line by line into files
for i, d in enumerate(twitter_data):
    with open(f"data_db/{i}.json", "w") as f:
        f.write(json.dumps(d) + "\n")

os.environ['OPENAI_API_KEY'] = "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"
os.environ['ACTIVELOOP_TOKEN'] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4NjkzOTc1NSwiZXhwIjoxNjg3MDI2MTQ3fQ.eyJpZCI6Ind1aWxsb3VkIn0.4L38izwLMktUsCyCkt76QatptwWsDcM-XZp7_yI-BgjTAI07PAPe1bhmbqm987QQ0u0KYrZIkhqrSaCT03vRTg"
# https://app.activeloop.ai/login
# https://app.activeloop.ai/wuilloud/langchain-code

root_dir = "/home/jair/WORK/TwitterBot/data_db/" #""datatest" #

print("###########")

# build file list
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.json'): # and '/.venv/' not in dirpath:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                pass
print(f'{len(docs)}')

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
from langchain.vectorstores import DeepLake


DEEPLAKE_ACCOUNT_NAME = "wuilloud"
db = DeepLake(dataset_path=f"hub://{DEEPLAKE_ACCOUNT_NAME}/langchain-code", read_only=True, embedding_function=embeddings)

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

n_k = 10
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = n_k
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = n_k

model = ChatOpenAI(model_name='gpt-3.5-turbo') # 'ada' 'gpt-3.5-turbo' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)


def search_database(question, chat_history=[]):
    print("question search database:: ", question)
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    result = qa({"question": question, \
                 "chat_history": chat_history})
    return result["answer"]


def hash_tag_analyser(txt):
    
    hash_tag_str = "summarise the meaning of the hashtag: "
    hash_tags = [s for s in txt.split() if s[0] == "#" and len(s)>1]
    
    utterance = "\n"    
    for hash_tag in hash_tags:
        utterance += search_database(hash_tag_str + hash_tag) + "\n"
    
    return utterance

def date_analyser(txt):
    
    analysis = "when and what was something very similar said: \n"+txt
    return search_database(analysis) + "\n"

def content_analyser(txt):
    
    analysis = "summarise what you find similar to: \n"+txt
    return search_database(analysis) + "\n"


#####################################################
#####################################################


####################################################
##### APP        ###################################
####################################################

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

    # take a random persona from list:
    persona = random.choice(personas)
    mssg = persona
    # add history to mssg
    for i in range(n_memory):
        if len(chat_history) > i:
            mssg += "\n" + chat_history[-i-1]["user"]
            mssg += "\n" + chat_history[-i-1]["agent"]
    # hash tag analyser
    mssg += hash_tag_analyser(user_input)
    # date analyser
    mssg += date_analyser(user_input)
    # content analyser
    mssg += content_analyser(user_input)

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
    return response_text, computation_time, price


if __name__ == '__main__':
    app.run(debug=True)

