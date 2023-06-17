
import os

from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


def build_deep_lake_db(root_dir=deeplake_local_path):

    root_dir = deeplake_local_path

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

    # process docs
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # build embeddings
    embeddings = OpenAIEmbeddings()
    # build database 
    db = DeepLake(dataset_path=f"hub://{deeplake_account_name}/langchain-code", read_only=True, embedding_function=embeddings)
    return db

def get_retriever(db, search_nk=deeplake_search_nk):
    """Returns a retriever with the given search_nk"""
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = search_nk
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = search_nk
    return retriever

def get_qa(model, deeplake_local_path):
    """Returns a ConversationalRetrievalChain with the given retriever"""
    db = build_deep_lake_db(deeplake_local_path)
    retriever = get_retriever(db)
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    return qa

def search_database(model, question, chat_history=[]):
    """Returns the answer from the database for the given question"""
    print("question search database:: ", question)
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    result = qa({"question": question, \
                 "chat_history": chat_history})
    return result["answer"]

def hash_tag_analyser(model, txt):
    """Identify hash tags and return summary of meaning from database"""
    hash_tag_str = "summarise the meaning of the hashtag: "
    hash_tags = [s for s in txt.split() if s[0] == "#" and len(s)>1]
    
    utterance = "\n"    
    for hash_tag in hash_tags:
        utterance += search_database(model, hash_tag_str + hash_tag) + "\n"
    
    return utterance

def date_analyser(model, txt):
    """Find dates of very similar utterances were mentionned in the database"""
    analysis = "when and what was something very similar said: \n"+txt
    return search_database(analysis) + "\n"

def content_analyser(model, txt):
    """Find content of very similar utterances were mentionned in the database"""
    analysis = "summarise what you find similar to: \n"+txt
    return search_database(analysis) + "\n"
