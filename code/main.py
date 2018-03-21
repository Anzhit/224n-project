# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import pickle
import logging

import tensorflow as tf
import numpy as np

from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json_data, generate_answers, save_answer_probs, get_batch_generator, generate_answers_with_bidaf

from nltk.tokenize.moses import MosesDetokenizer


logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 128, "Size of the hidden states")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers")
tf.app.flags.DEFINE_integer("context_len", 450, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

# Misc
tf.app.flags.DEFINE_boolean("cudnn_lstm", True, "Whether to use cudnn_lstm")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print "Looking for model at %s..." % train_dir
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
#            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())
    
    model.feed_embedding(session)


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Print out Tensorflow version
    print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    # Define train_dir
    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    if FLAGS.mode != 'loadProbs':
        # Load embedding matrix and vocab mappings
        emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

        # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
        train_context_path = os.path.join(FLAGS.data_dir, "train.context")
        train_qn_path = os.path.join(FLAGS.data_dir, "train.question")
        train_ans_path = os.path.join(FLAGS.data_dir, "train.span")
        dev_context_path = os.path.join(FLAGS.data_dir, "dev.context")
        dev_qn_path = os.path.join(FLAGS.data_dir, "dev.question")
        dev_ans_path = os.path.join(FLAGS.data_dir, "dev.span")

        # Initialize model
        qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

        # Some GPU settings
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train":

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        # with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            # json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:

            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)


    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=20, print_to_screen=True)


    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print "Wrote predictions to %s" % FLAGS.json_out_path
    
    elif FLAGS.mode == "official_eval_with_bidaf":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict, bidaf_dict, self_dict1, self_dict2, out_dict = generate_answers_with_bidaf(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print "Wrote predictions to %s" % FLAGS.json_out_path

            print "Writing sims to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path + '-bidaf', 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(bidaf_dict, ensure_ascii=False)))
                print "Wrote sims to %s" % FLAGS.json_out_path

            print "Writing self sims1 to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path + '-self1', 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(self_dict1, ensure_ascii=False)))
                print "Wrote self sims1 to %s" % FLAGS.json_out_path

            print "Writing self sims2 to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path + '-self2', 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(self_dict2, ensure_ascii=False)))
                print "Wrote self sims2 to %s" % FLAGS.json_out_path
                
            print "Writing preds to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path + '-preds', 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(out_dict, ensure_ascii=False)))
                print "Wrote preds to %s" % FLAGS.json_out_path

                
    elif FLAGS.mode == 'saveProbs':
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict = save_answer_probs(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'wb') as f:
                pickle.dump(answers_dict, f, protocol=2)
                # f.write(unicode(pickle.dumps(answers_dict, ensure_ascii=False)))
                # f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print "Wrote predictions to %s" % FLAGS.json_out_path
                
    elif FLAGS.mode == 'loadProbs':
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)
        word2id = pickle.load(open('word2id', 'rb'))
#         pickle.dump(word2id, open('word2id', 'wb'))
        print 'Loaded data'
        
        dictLists = []
        for file in os.listdir('./pickles'):
            f = os.path.join('./pickles', file)
            print 'Loading predictions from ', f
            prob_dict = pickle.load(open(f, 'rb'))
            dictLists += [prob_dict]
        
#         mainDict = {}
#         stdiDict = {}
#         for probs in dictLists:
#             for k in dictLists[0].keys():
#                 stdi = 1.0 / (np.std(np.array(probs[k][0])) + np.std(np.array(probs[k][1])) + 1e-2)
#                 stdiDict[k] = stdi
#                 try:
#                     mainDict[k] = (mainDict[k][0] + stdi * np.array(probs[k][0]), mainDict[k][1] + stdi* np.array(probs[k][1]))
#                 except KeyError:
#                     mainDict[k] = (stdi* np.array(probs[k][0]), stdi*np.array(probs[k][1]))
            
        uuid2ans = {} # maps uuid to string containing predicted answer
        detokenizer = MosesDetokenizer()
        
#         for k in mainDict.keys():
#             start_dist = mainDict[k][0] / stdiDict[k]
#             end_dist = mainDict[k][1] / stdiDict[k]
            
#             # Take argmax to get start_pos and end_post, both shape (batch_size)
#             end_dp = np.zeros(end_dist.shape)
# #             start_pos = np.argmax(start_dist)
# #             end_pos = np.argmax(end_dist)
#             end_dp[-1]=end_dist[-1]
#             for i in range(len(end_dist)-2,-1,-1):
#                 end_dp[i]=np.amax([end_dist[i],end_dp[i+1]])
#             start_pos=np.argmax(start_dist*end_dp)
#             end_pos = start_pos + np.argmax(end_dist[start_pos:])
            
#             uuid2ans[k] = (start_pos, end_pos)
        
        for k in dictLists[0].keys():
            spanDict = {}
            for probs in dictLists:
                start_dist = np.array(probs[k][0])
                end_dist = np.array(probs[k][1])

                # Take argmax to get start_pos and end_post, both shape (batch_size)
                end_dp = np.zeros(end_dist.shape)
                end_dp[-1]=end_dist[-1]
                for i in range(len(end_dist)-2,-1,-1):
                    end_dp[i]=np.amax([end_dist[i],end_dp[i+1]])
                start_pos=np.argmax(start_dist*end_dp)
                end_pos = start_pos + np.argmax(end_dist[start_pos:])

                try:
                    spanDict[(start_pos, end_pos)] += [start_dist[start_pos] * end_dist[end_pos]]
                except KeyError:
                    spanDict[(start_pos, end_pos)] = [start_dist[start_pos] * end_dist[end_pos]]

            best_span = (0, 0)
            best_span_votes = 0
            best_span_prob = 0
            for span in spanDict.keys():
                if len(spanDict[span]) > best_span_votes:
                    best_span = span
                    best_span_votes = len(spanDict[span])
                    best_span_prob = max(spanDict[span])
                elif len(spanDict[span]) == best_span_votes and best_span_prob < max(spanDict[span]):
                    best_span = span
                    best_span_votes = len(spanDict[span])
                    best_span_prob = max(spanDict[span])

            uuid2ans[k] = (best_span[0], best_span[1])

        result = {}
        data_size = len(qn_uuid_data)
        num_batches = ((data_size-1) / FLAGS.batch_size) + 1
        batch_num = 0
        print "Generating answers..."
        
        for batch in get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, FLAGS.batch_size, FLAGS.context_len, FLAGS.question_len):

            # For each example in the batch:
            for ex_idx in range(FLAGS.batch_size):

                # Detokenize and add to dict
                try:    
                    uuid = batch.uuids[ex_idx]
                    pred_start, pred_end = uuid2ans[uuid]

                    # Original context tokens (no UNKs or padding) for this example
                    context_tokens = batch.context_tokens[ex_idx] # list of strings

                    # Check the predicted span is in range
                    assert pred_start in range(len(context_tokens))
                    assert pred_end in range(len(context_tokens))

                    # Predicted answer tokens
                    pred_ans_tokens = context_tokens[pred_start : pred_end +1] # list of strings

                    result[uuid] = detokenizer.detokenize(pred_ans_tokens, return_str=True)

                except IndexError:
                    pass

            batch_num += 1

            if batch_num % 10 == 0:
                print "Generated answers for %i/%i batches = %.2f%%" % (batch_num, num_batches, batch_num*100.0/num_batches)

        print "Finished generating answers for dataset."
        answers_dict = result

        # Write the uuid->answer mapping a to json file in root dir
        print "Writing predictions to %s..." % FLAGS.json_out_path
        with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
            print "Wrote predictions to %s" % FLAGS.json_out_path
            

    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
