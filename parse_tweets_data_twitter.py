
import sys
import json


def parse_tweets_data(tweets_data_path, twitter_handle):

    # open file and read data line by line
    with open(tweets_data_path) as f:
        data = f.readlines()
    
    data = [json.loads(line[:-1]) for line in data]

    # map data list into a list of dictionaries in format:
    # {"prompt": "tweet", "completion": "retweet"}

    # initialize list of dictionaries
    data_list = []

    # iterate through data list
    for i in range(len(data)):
        # initialize dictionary
        data_dict = {}
        # add prompt/retweet
        if "retweeted_status" in data[i]:
            data_dict["completion"] = data[i]["text"]
            retweet_data = data[i]["retweeted_status"]
            data_dict["prompt"] = retweet_data["text"] 
            # + " screen_name: " + retweet_data["user"]["screen_name"] + " retweet_id: " + retweet_data["user"]["screen_name"]
        else:
            data_dict["completion"] = data[i]["text"]
            data_dict["prompt"] = "generate a random tweet from " + twitter_handle
        # add dictionary to list
        data_list.append(data_dict)

    return data_list


# code to reads argument tweets_data_path, which is a string
# and calls parse_tweets_data on tweets_data_path
# and prints the result
if __name__ == "__main__":

    # extract parameter from command line    
    if len(sys.argv) != 2:
        print("Usage: python parse_tweets_data.py <twitter_handle>")
        sys.exit(1)
    twitter_handle = sys.argv[1]

    # set paths
    tweets_data_path = "data/twitter_data_"+twitter_handle+".jsonl"
    tweets_data_path_out = "/".join(tweets_data_path.split("/")[:-1]) + "/parsed_" + tweets_data_path.split("/")[-1]

    # parse tweets_data
    tweets_data = parse_tweets_data(tweets_data_path, twitter_handle)

    # write tweets_data to file line by line
    with open(tweets_data_path_out, "w") as f:
        for item in tweets_data:
            f.write(json.dumps(item) + '\n')
