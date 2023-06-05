# TwitterBot

Automatised synthetic tweet generation from a real twitter handle.

The bot can be trained generating synthetic data from well known twitter user (DonaldTrum, ...) 
or alternatively, real twitter data can be used.


## Run 
update config.yaml and run:

`python api.py`

It will run on http://127.0.0.1:5000. Open that url and your browser.


To generate **tweets**, enter generate a random tweet from {twitter_handle}, with twitter_handle that is for instance the realDonaldTrump or alternatively just send an empy string (keep prompt on webpage empty en click send).


To generate **retweets**, enter random tweets the agent should be answering/retweeting to. 


### Requirements
One needs a openai key to generate the synthetic data and twitter secret token, access key and so on to download twitter data to built a twitterbot from it.


### Configuration
Meta data and parameters to run all functionalities of TwitterBot 
./config.yaml


### Installation
Use the standard requirements.txt file. 

Alternatively, installing the libraries below should be enough:
`
pip install --upgrade openai
pip install names
pip install PyYAML
pip install langchain
pip install flask
pip install sentence_transformers
pip install tqdm
pip install pickle5
`


### Scripts

The model are built through the following scripts.

#### Generate And Massage Twitter Data
Two modes are supported:

1. **Mock Twitter Data**
Use a wellknown twitter handle in the config file and run:

`bash script_generate_mock_tweets.sh`

2. **Download Real Twitter Data**
This workflow has never been tested and is expected to require slight adjustments:

`bash script_download_and_preprocess_twitter_data.sh`


#### Prepare Data and Train Model
Allows to get fine_tune_job_id

`bash script_prepare_data_and_train_model.sh`


#### Monitor Model
Requires fine_tune_job_id in config.

`bash script_monitor_openai_agent.sh`

Training is over when all training epochs have been ran and model gets a "fine_tuned_model" name.