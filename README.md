# TwitterBot

Automatised synthetic tweet generation from a real twitter handle.

The bot can be trained generating synthetic data from well known twitter user (DonaldTrum, ...) 
or alternatively, real twitter data can be used.


## Run 
update config.yaml and run:

`python api.py`


### Requirements
One needs a openai key to generate the synthetic data and twitter secret token, access key and so on to download twitter data to built a twitterbot from it.


### Configuration

Meta data and parameters to run all functionalities of TwitterBot 
./config.yaml


### Scripts

The model are built through the following scripts.

#### Generate And Massage Twitter Data
Two modes are supported:

1. **Mock Twitter Data**
Use a wellknown twitter handle in the config file and run:

`bash script_generate_mock_tweets.sh`

2. **Download Real Twitter Data**
This workflow has never been tested and is expected to require adjustment:

`bash script_download_and_preprocess_twitter_data.sh`


#### Prepare Data and Train Model
Allows to get fine_tune_job_id

`bash script_prepare_data_and_train_model.sh`


#### Monitor Model
Requires fine_tune_job_id in config.

`bash script_monitor_openai_agent.sh`

Training is over when all training epochs have been ran and model gets a "fine_tuned_model" name.