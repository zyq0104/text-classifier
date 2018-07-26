
import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf

# tf.flags.DEFINE_string("label1_datas", "./train_label1_data_with_slot", "Data source for the label1 data.")
# tf.flags.DEFINE_string("label2_datas", "./train_label2_data_with_slot", "Data source for the label2 data.")
# tf.flags.DEFINE_string("label3_datas", "./train_label3_data_with_slot", "Data source for the label3 data.")

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"？", "", string)
    string = re.sub(r"！", "", string)
    string = re.sub(r"，", "", string)
    string = re.sub(r"。", "", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def load_data_and_labels(label1
_datas, label2_datas, label3_datas, food_dict = None, module = True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    label1
_examples = list(open(label1
    _datas, "r", encoding="utf-8").readlines())
    label1
_examples = [s.strip() for s in label1
_examples]
    label2_examples = list(open(label2_datas, "r", encoding="utf-8").readlines())
    label2_examples = [s.strip() for s in label2_examples]
    label3_examples = list(open(label3_datas, "r", encoding="utf-8").readlines())
    label3_examples = [s.strip() for s in label3_examples]

    # label1
_example = []
    # label1
_example_slot = []
    # label2_example = []
    # label2_example_slot = []
    # label3_example = []
    # label3_example_slot = []
    #
    # for i in label1
_examples:
    #     # print(i)
    #     # print(i.split("@"))
    #     split = i.split("@")
    #     # print(split[0])
    #     label1
_example.append(split[0])
    #     label1
_example_slot.append(split[1])
    # # print(label1
_example_slot[2])
    #
    # for i in label2_examples:
    #     # print(i)
    #     # print(i.split("@"))
    #     split = i.split("@")
    #     # print(split[0])
    #     label2_example.append(split[0])
    #     label2_example_slot.append(split[1])
    #
    # for i in label3_examples:
    #     # print(i)
    #     # print(i.split("@"))
    #     split = i.split("@")
    #     # print(split[0])
    #     label3_example.append(split[0])
    #     label3_example_slot.append(split[1])


    if(food_dict != None):
        food_dict = list(open(food_dict, "r").readlines())
        food_dict_example = [s.strip() for s in food_dict]
    else:
        food_dict_example = []
    # Split by words
    x_text = food_dict_example + label1
_examples + label2_examples + label3_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_label1
 = label1
    _examples + food_dict_example
    x_label2 = label2_examples
    x_label3 = label3_examples
    x_label1
 = [clean_str(sent) for sent in x_label1
    ]
    x_label2 = [clean_str(sent) for sent in x_label2]
    x_label3 = [clean_str(sent) for sent in x_label3]
    # x_label1
_slot = label1
_example_slot
    # x_label2_slot = label2_example_slot
    # x_label3_slot = label3_example_slot
    # x_label1
_slot = [clean_str(sent) for sent in x_label1
_slot]
    # x_label2_slot = [clean_str(sent) for sent in x_label2_slot]
    # x_label3_slot = [clean_str(sent) for sent in x_label3_slot]
    # x_text_slot = x_label1
_slot + x_label2_slot + x_label3_slot

    # Generate labels
    label1
_labels = [[1, 0, 0] for _ in label1
_examples]
    label2_labels = [[0, 1, 0] for _ in label2_examples]
    label3_labels = [[0, 0, 1] for _ in label3_examples]
    # print(len(label3_labels))
    y = np.concatenate([label1
    _labels, label2_labels, label3_labels], axis=0)
    # print(x_text)
    if module:
        return [x_text, y]
    else:
        return [x_text, x_label1
    , x_label2, x_label3, y, label1
    _labels, label2_labels, label3_labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# load_data_and_labels(FLAGS.label1_datas, FLAGS.label2_datas, FLAGS.label3_datas)