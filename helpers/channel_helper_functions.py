import pickle
from FlowCytometryTools import FCMeasurement
import os

# ******************************** ONE TIME FUNCTIONS *******************************#
# Create pickle to store Dictionary of channels - one time use


def create_index_channel_dict(channels):
    channels_dic = {}
    for idx,channel in zip(range(1,len(channels)+1),channels):
        channels_dic[idx] = channel
    channel_dict_file = "index_channel_file"
    fileObject = open(channel_dict_file,'wb')
    pickle.dump(channels_dic,fileObject)
    fileObject.close()


# Dont use this function - - one time use
def get_channels_names_from_file(data_dir):
    some_fcs_file = os.listdir(data_dir)[0]
    file_name_path = os.path.join(data_dir, some_fcs_file)
    human = FCMeasurement(ID='Test Sample', datafile=file_name_path)
    channels = human.channel_names[2:]
    all_channels_names = [str(channels[f]) for f in range(len(channels))]
    return all_channels_names

# ************************************************************************************#


# Returns all channels name in a list
def get_channels_names_from_pickle():
    fileObject = open("index_channel_file",'r')
    dict = pickle.load(fileObject)
    fileObject.close()
    return dict.values()


# Map Channel to its index
def channel_to_index(channel_name):
    fileObject = open("index_channel_file",'r')
    dict = pickle.load(fileObject)
    fileObject.close()
    for k,v in dict.iteritems():
        if v == channel_name:
            return k
    print "Error: no channel found"

# Map Index to its channel
def index_to_channel(channel_index):
    fileObject = open("index_channel_file",'r')
    dict = pickle.load(fileObject)
    fileObject.close()
    return dict[channel_index]


