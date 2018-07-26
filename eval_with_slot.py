#! /usr/bin/env python

import tensorflow as tf
import numpy as np
from collections import Counter
import os
import data_helpers
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("label1_datas", "./test_datas_label1_final", "Data source for the label1 data.")
# tf.flags.DEFINE_string("label2_datas", "./test_datas_label2_final", "Data source for the label2 data.")
# tf.flags.DEFINE_string("label3_datas", "./test_datas_label3_final", "Data source for the label3 data.")

# tf.flags.DEFINE_string("train_label1_datas", "./data_label1_new", "Data source for the label1 data.")
# tf.flags.DEFINE_string("train_label2_datas", "./data_label2_new", "Data source for the label2 data.")
# tf.flags.DEFINE_string("train_label3_datas", "./data_label3_new", "Data source for the label3 data.")


tf.flags.DEFINE_string("label1_datas", "./final_2/data_label1_test", "Data source for the label1 data.")
tf.flags.DEFINE_string("label2_datas", "./final_2/data_label2_test", "Data source for the label2 data.")
tf.flags.DEFINE_string("label3_datas", "./final_2/data_label3_test", "Data source for the label3 data.")
tf.flags.DEFINE_integer("num_slot_dim", 18, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")
tf.flags.DEFINE_integer("slot_1", 15, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")
tf.flags.DEFINE_integer("slot_0", 10, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1523649466/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("Eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.Eval_train:
    x_raw, x_label1, x_label2, x_label3, y_test, y_label1, y_label2, y_label3 = \
        data_helpers.load_data_and_labels(FLAGS.label1_datas, FLAGS.label2_datas, FLAGS.label3_datas, module=False)
    y_test = np.argmax(y_test, axis=1)
    y_label1 = np.argmax(y_label1, axis=1)
    y_label2 = np.argmax(y_label2, axis=1)
    y_label3 = np.argmax(y_label3, axis=1)
else:
    x_raw = ["我要金百万.", "昨天定的那一单"]
    y_test = [0, 1]

# x_train, y_trian = data_helpers.load_data_and_labels(FLAGS.train_label1_datas, FLAGS.train_label2_datas, FLAGS.train_label3_datas)
# print(wordlist.shape)

x_rem_label1 = x_label1
x_rem_label2 = x_label2
x_rem_label3 = x_label3
# Map data into vocabulary

voca = []
voca_slot = []
for x in x_raw:
    x_split = str(x).split("@")
    spli_str = ""
    for i in x_split[0]:
        spli_str = spli_str + " " + i
    voca.append(spli_str)
    voca_slot.append(x_split[1])

voca_slot_1 = []
for i in voca_slot:
    i = i.strip("[")
    i = i.strip("]")
    i = i.split(" ")
    # print(i)
    i = [int(x) for x in i]
    # print(len(i))
    voca_slot_1.append(i)
voca_slot = voca_slot_1

# voca = []
#
# for x in x_raw:
#     spli_str = ""
#     for i in x:
#         spli_str = spli_str + " " + i
#     voca.append(spli_str)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(voca)))

print(np.array(voca_slot).shape)
# for i in voca_slot:
#     if FLAGS.slot_1 in i:
#         i.append(1)
#     else:
#         i.append(0)
print(np.array(x_test).shape)
x_test = np.concatenate([x_test, np.array(voca_slot)], axis=1)
print(x_test.shape)

voca = []
voca_slot = []
for x in x_label1:
    x_split = str(x).split("@")
    spli_str = ""
    for i in x_split[0]:
        spli_str = spli_str + " " + i
    voca.append(spli_str)
    voca_slot.append(x_split[1])

voca_slot_1 = []
for i in voca_slot:
    i = i.strip("[")
    i = i.strip("]")
    i = i.split(" ")
    # print(i)
    i = [int(x) for x in i]
    # print(len(i))
    voca_slot_1.append(i)
voca_slot = voca_slot_1
# for x in x_label1:
#     spli_str = ""
#     for i in x:
#         spli_str = spli_str + " " + i
#     voca.append(spli_str)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_label1 = np.array(list(vocab_processor.transform(voca)))

print(np.array(voca_slot).shape)
# for i in voca_slot:
#     if FLAGS.slot_1 in i:
#         i.append(1)
#     else:
#         i.append(0)
x_label1 = np.concatenate([x_label1, np.array(voca_slot)], axis=1)
print(x_label1.shape)


voca = []
voca_slot = []
for x in x_label2:
    x_split = str(x).split("@")
    spli_str = ""
    for i in x_split[0]:
        spli_str = spli_str + " " + i
    voca.append(spli_str)
    voca_slot.append(x_split[1])

voca_slot_1 = []
for i in voca_slot:
    i = i.strip("[")
    i = i.strip("]")
    i = i.split(" ")
    # print(i)
    i = [int(x) for x in i]
    # print(len(i))
    voca_slot_1.append(i)
voca_slot = voca_slot_1

# voca = []
# for x in x_label2:
#     spli_str = ""
#     for i in x:
#         spli_str = spli_str + " " + i
#     voca.append(spli_str)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_label2 = np.array(list(vocab_processor.transform(voca)))

print(np.array(voca_slot).shape)
# for i in voca_slot:
#     if FLAGS.slot_1 in i:
#         # print(1)
#         i.append(1)
#     else:
#         i.append(0)
x_label2 = np.concatenate([x_label2, np.array(voca_slot)], axis=1)
print(x_label2.shape)


voca = []
voca_slot = []
for x in x_label3:
    x_split = str(x).split("@")
    spli_str = ""
    for i in x_split[0]:
        spli_str = spli_str + " " + i
    voca.append(spli_str)
    voca_slot.append(x_split[1])

voca_slot_1 = []
for i in voca_slot:
    i = i.strip("[")
    i = i.strip("]")
    i = i.split(" ")
    # print(i)
    i = [int(x) for x in i]
    # print(len(i))
    voca_slot_1.append(i)
voca_slot = voca_slot_1

voca = []
for x in x_label3:
    spli_str = ""
    for i in x:
        spli_str = spli_str + " " + i
    voca.append(spli_str)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_label3 = np.array(list(vocab_processor.transform(voca)))

print(np.array(voca_slot).shape)
# for i in voca_slot:
#     if FLAGS.slot_1 in i:
#         i.append(1)
#     else:
#         i.append(0)
x_label3 = np.concatenate([x_label3, np.array(voca_slot)], axis=1)
print(x_label3.shape)

print(x_test)
print("\nEvaluating...\n")


# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print(checkpoint_file)
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        slot_value = graph.get_operation_by_name("slot").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # zero_num = graph.get_operation_by_name("zero_num").outputs[0]
        # embedding_W = graph.get_operation_by_name("embedding/embedding_W").outputs[0]
        # embedded_chars = graph.get_operation_by_name("embedding/embedded_chars").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        score = graph.get_operation_by_name("output/scores").outputs[0]
        drop = graph.get_operation_by_name("pool").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        bathces_label1 = data_helpers.batch_iter(list(x_label1), FLAGS.batch_size, 1, shuffle=False)
        bathces_label2 = data_helpers.batch_iter(list(x_label2), FLAGS.batch_size, 1, shuffle=False)
        bathces_label3 = data_helpers.batch_iter(list(x_label3), FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        label1_pre = []
        label2_pre = []
        label3_pre = []
        for x_test_batch in batches:
            batch_predictions, sco = sess.run([predictions, score], {input_x: x_test_batch[:, :97],
                                                                     slot_value: x_test_batch[:, 97:],
                                                                     dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        for x_test_batch_label1 in bathces_label1:
            batch_predictions_label1, sco_label1 = sess.run([predictions, score], {input_x: x_test_batch_label1[:, :97],
                                                                                 slot_value: x_test_batch_label1[:, 97:],
                                                                                 dropout_keep_prob: 1.0})
            label1_pre = np.concatenate([label1_pre, batch_predictions_label1])
            # print(sco_label1.shape)

        for x_test_batch_label2 in bathces_label2:
            batch_predictions_label2, sco_label2 = sess.run([predictions, score], {input_x: x_test_batch_label2[:, :97],
                                                                                     slot_value: x_test_batch_label2[:, 97:],
                                                                                     dropout_keep_prob: 1.0})
            label2_pre = np.concatenate([label2_pre, batch_predictions_label2])

        for x_test_batch_label3 in bathces_label3:
            batch_predictions_label3, sco_label3 = sess.run([predictions, score], {input_x: x_test_batch_label3[:, :97],
                                                                                 slot_value: x_test_batch_label3[:, 97:],
                                                                                 dropout_keep_prob: 1.0})
            label3_pre = np.concatenate([label3_pre, batch_predictions_label3])

# Print accuracy if y_test is defined
label1_ri = 0
label2_ri = 0
label3_ri = 0
if y_test is not None:
    print(all_predictions)
    output = open("test_reply", "a", encoding="UTF-8")
    for i in range(len(all_predictions)):
        # print(x_raw[i], y_test[i])
        # print(all_predictions[i])
        output.write(x_raw[i].split("@")[0])
        output.write(" ")
        output.write(str(all_predictions[i]))
        output.write("\n")
    correct_predictions = float(sum(all_predictions == y_test))


    for i in range(len(all_predictions)):
        if y_test[i] == 0.0 and all_predictions[i] == float(y_test[i]):
            label1_ri += 1
        if y_test[i] == 1.0 and all_predictions[i] == float(y_test[i]):
            label2_ri += 1
        if y_test[i] == 2.0 and all_predictions[i] == float(y_test[i]):
            label3_ri += 1
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

count = Counter(all_predictions)
if count[1.0] == 0: count[1.0] = 1
if count[2.0] == 0: count[2.0] = 1

if y_label1 is not None:
    #print(len(label1_pre))
    #print(len(y_label1))
    correct_predictions = float(sum(label1_pre == y_label1))
    # for i in range(len(y_label1)):
    #      if(label1_pre[i] != y_label1[i]):
    #          print(x_rem_label1[i])
    #          print(label1_pre[i])
    print("Total number of label1 examples: {}".format(len(y_label1)))
    # print("Accuracy of label1: {:g}".format(correct_predictions/float(len(y_label1))))
    print("Precision of label1: {:g}".format(label1_ri/float(count[0.0])))
    print("Recall of oreder: {:g}".format(label1_ri/float(len(y_label1))))

if y_label2 is not None:
    # print(label2_pre)
    # for i in all_predictions:
    #      print(i)
    correct_predictions = float(sum(label2_pre == y_label2))
    # for i in range(len(y_label2)):
    #      if(label2_pre[i] != y_label2[i]):
    #          print(x_rem_label2[i])
    #          print(label2_pre[i])
    print("Total number of label2 examples: {}".format(len(y_label2)))
    # print("Accuracy: {:g}".format(correct_predictions/float(len(y_label2))))
    print("Precision of label2: {:g}".format(label2_ri / float(count[1.0])))
    print("Recall of reoreder: {:g}".format(label2_ri / float(len(y_label2))))

if y_label3 is not None:
    #print(label3_pre)
    # for i in all_predictions:
    #      print(i)
    correct_predictions = float(sum(label3_pre == y_label3))
    # for i in range(len(y_label3)):
    #      if(label3_pre[i] != y_label3[i]):
    #          print(x_rem_label3[i])
    #          print(label3_pre[i])
    print("Total number of label3 examples: {}".format(len(y_label3)))
    # print("Accuracy: {:g}".format(correct_predictions/float(len(y_label3))))
    print(count[2.0])
    print("precision of label3: {:g}".format(label3_ri / float(count[2.0])))
    print("recall of label3: {:g}".format(label3_ri / float(len(y_label3))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', encoding="utf-8") as f:
    csv.writer(f).writerows(predictions_human_readable)
