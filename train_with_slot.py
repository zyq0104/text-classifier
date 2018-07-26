#! /usr/bin/env python
# encoding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn_modify import TextCNN
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_float("test_sample_percentage", .0, "Percentage of the training data to use for test")
# tf.flags.DEFINE_float("train_sample_percentage", .9, "Percentage of the training data to use for train")
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("label1_datas", "./final_2/data_label1_train", "Data source for the label1 data.")
tf.flags.DEFINE_string("label2_datas","./final_2/data_label2_train", "Data source for the label2 data.")
tf.flags.DEFINE_string("label3_datas", "./final_2/data_label3_train", "Data source for the label3 data.")
tf.flags.DEFINE_string("food_dict", "./food_dict", "food dictionary")
tf.flags.DEFINE_string("embedded_word2vec", "./food_dict", "food dictionary")
# tf.flags.DEFINE_string("test_label1_datas", "./test_data_label1_new", "Data source for the label1 data.")
# tf.flags.DEFINE_string("test_label2_datas", "./test_data_label2_new", "Data source for the label2 data.")
# tf.flags.DEFINE_string("test_label3_datas", "./test_data_label3_new", "Data source for the label3 data.")

# tf.flags.DEFINE_string("label1_datas", "./train_label1_data_with_slot", "Data source for the label1 data.")
# tf.flags.DEFINE_string("label2_datas", "./train_label2_data_with_slot", "Data source for the label2 data.")
# tf.flags.DEFINE_string("label3_datas", "./train_label3_data_with_slot", "Data source for the label3 data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularization lambda (default: 0.0)")
# tf.flags.DEFINE_integer("slot_1", 15, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")
# tf.flags.DEFINE_integer("slot_0", 10, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_slot_dim", 18, "slot dim(poi_name, product_name, etc) , default:37 decided by NER model")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.label1_datas, FLAGS.label2_datas, FLAGS.label3_datas)
# x_test_test, y_test = data_helpers.load_data_and_labels(FLAGS.test_label1_datas, FLAGS.test_label2_datas,
#                                                         FLAGS.test_label3_datas)
# print(y)
# for x in x_text:
#     print(len(x))
# Build vocabulary
voca = []
voca_slot = []
for x in x_text:
    x_split = str(x).split("@")
    spli_str = ""
    for i in x_split[0]:
        spli_str = spli_str + " " + i
    voca.append(spli_str)
    voca_slot.append(x_split[1])
    # (x_split[0])

voca_slot_1 = []
for i in voca_slot:
    i = i.strip("[")
    i = i.strip("]")
    i = i.split(" ")
    # print(i)
    i = [int(x) for x in i]
    # print(len(i))
    voca_slot_1.append(i)
# print(np.array(voca_slot_1).shape)
voca_slot = voca_slot_1

len_vo = []
for i in voca:
    list_i = i.strip("").split(" ")
    #print(list_i)
    len_vo = np.concatenate([len_vo, [len(list_i)]])
max_document_length = max(len_vo)
print(max_document_length)
max_document_length = int(max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(voca)))

print(np.array(voca_slot).shape)
# for i in voca_slot:
#     if FLAGS.slot_1 in i:
#         i.append(1)
#     else:
#         i.append(0)
print(np.array(voca_slot).shape)
x = np.concatenate([x, np.array(voca_slot)], axis=1)
# print(x)
# for i in x:
#     print(i)

# for i in x_test_test:
#     spli_str = ""
#     for j in i:
#         spli_str = spli_str + " " + j
#     voca.append(spli_str)

# Get trained embedding layer(gensim)
# wordlist = embedding_W.embedding_W(x_text, x_test_test)
# print(wordlist.shape)


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/validation set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# for i in y_train:
#     print(i)
#print(y_dev)
del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/dev: {:d}/{:d}".format(len(y_train), len(y_dev)))

print(x_dev.shape)
print(x_train.shape)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_document_length,
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            slot_dim=FLAGS.num_slot_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary

        vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))
        print("save vocab in " + os.path.join(checkpoint_dir))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            # print(x_batch)
            x = x_batch[:, max_document_length:]
            x_zero = np.zeros(shape=[x.shape[0],  x.shape[0]])
            number = 0
            for i in x:
                x_num = np.count_nonzero(i)
                if x_num != 0:
                    x_zero[number][number] = 1
                number+=1

            feed_dict = {
                # cnn.input_x: x_batch[:, :max_document_length],
                cnn.input_x:x_batch[:,:max_document_length],
                cnn.input_y: y_batch,
                cnn.slot: x_batch[:, max_document_length:],
                cnn.zero_num: x_zero,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              # cnn.embedding_W: wordlist
            }
            # print(word2vec_list)
            _, iy, sco, step, summaries, loss, accuracy = sess.run(
                [train_op, cnn.labels, cnn.predictions, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            # sess.run(tf.Print(cnn.embedded_chars, [cnn.embedded_chars], summarize=15),feed_dict)
            # print(iy)
            # print(sco)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            x = x_batch[:, max_document_length:]
            x_zero = np.zeros(shape=[x.shape[0],  x.shape[0]])
            number = 0
            for i in x:
                x_num = np.count_nonzero(i)
                if x_num != 0:
                    x_zero[number][number] = 1
                number+=1
            feed_dict = {
                cnn.input_x: x_batch[:, :max_document_length],
                cnn.input_y: y_batch,
                cnn.slot: x_batch[:, max_document_length:],
                cnn.zero_num: x_zero,
                cnn.dropout_keep_prob: 1.0,
              # cnn.embedding_W: wordlist
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: stepx {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # model = w2v.Word2Vec()
        # model = gm.KeyedVectors.load_word2vec_format("vector.txt")
        # Training loop. For each batch...
        boo = True
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch)
            # if boo:
            #     out_put = open("test_1", "w")
            #     out_put.write(str(x_batch))
            #     out_put.write("\n")
            #     out_put.write(str(x_batch[:, :max_document_length]))
            #     out_put.write("\n")
            #     out_put.write(str(x_batch[:, max_document_length:]))
            #     boo = False
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                #test_step(x_test, y_test, checkpoint=checkpoint_dir, path=path)

