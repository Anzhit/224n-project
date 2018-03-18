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

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob, num_layers, use_cudnn_lstm=False, batch_size=None,cudnn_dropout=None):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.use_cudnn_lstm = use_cudnn_lstm
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.cudnn_dropout=cudnn_dropout
        if self.use_cudnn_lstm:
            print('Using cudnn lstm')
            self.direction = 'bidirectional'
            
            self.cudnn_cell = cudnn_rnn.CudnnLSTM(self.num_layers, self.hidden_size,
                                                          direction=self.direction,dropout=cudnn_dropout)
        else:
            self.rnn_cell_fw = [tf.contrib.rnn.LSTMCell(self.hidden_size, name='lstmf'+str(i)) for i in range(num_layers)]
            self.rnn_cell_fw = [DropoutWrapper(self.rnn_cell_fw[i], input_keep_prob=self.keep_prob) for i in range(num_layers)]
            self.rnn_cell_bw = [tf.contrib.rnn.LSTMCell(self.hidden_size, name='lstmb'+str(i)) for i in range(num_layers)]
            self.rnn_cell_bw = [DropoutWrapper(self.rnn_cell_bw[i], input_keep_prob=self.keep_prob) for i in range(num_layers)]

    def build_graph(self, inputs, masks, id='',is_training=None):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"+id):
            if self.use_cudnn_lstm:
                inputs = tf.transpose(inputs, [1, 0, 2])
                # params_size_t = self.cudnn_cell.count_params()
                # self.rnn_params = tf.get_variable("lstm_params", initializer=tf.random_uniform([params_size_t], -0.1, 0.1), validate_shape=False)
                # self.c = tf.zeros([self.num_layers, options.batch_size, self.hidden_size], tf.float32)
                # self.h = tf.zeros([self.num_layers, options.batch_size, self.hidden_size], tf.float32)
                outputs,_ = self.cudnn_cell(inputs,training= is_training)
                in_text_repres = tf.transpose(outputs, [1, 0, 2])
                in_text_repres = tf.nn.dropout(in_text_repres, self.keep_prob)
                return in_text_repres
            else: 
                input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
                out = inputs

                for i in range(self.num_layers):
                    # Note: fw_out and bw_out are the hidden states for every timestep.
                    # Each is shape (batch_size, seq_len, hidden_size).
                    (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw[i], self.rnn_cell_bw[i], out, input_lens, dtype=tf.float32)

                    # Concatenate the forward and backward hidden states
                    out = tf.concat([fw_out, bw_out], 2)

                    # Apply dropout
                    out = tf.nn.dropout(out, self.keep_prob)
                return out

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

        
class Bidaf(object):
    def __init__(self, keep_prob, context_vec_size, qn_vec_size):
        self.keep_prob = keep_prob
        self.context_vec_size = context_vec_size
        self.qn_vec_size = qn_vec_size

    def build_graph(self, context, context_mask, qns, qns_mask, scope):
        with vs.variable_scope(scope):

            # context:  (batch_size, context_len, hidden_size*2)
            # qns    :  (batch_size, question_len, hidden_size*2)
            
            assert self.context_vec_size == self.qn_vec_size
            ws1 = tf.get_variable("ws1",shape=[self.qn_vec_size,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            ws2 = tf.get_variable("ws2",shape=[self.qn_vec_size,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            ws3 = tf.get_variable("ws3",shape=[self.qn_vec_size,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            
            wh = tf.tensordot(context, ws1, [[2], [0]])   # (batch_size, context_len)
            wq = tf.tensordot(qns, ws2, [[2], [0]])       # (batch_size, question_len)
            woh = tf.multiply(context, ws3)               # (batch_size, context_len, hidden_size*2)
            qnsT = tf.transpose(qns, perm=[0, 2, 1])      # (batch_size, hidden_size*2, question_len)
            whoq = tf.matmul(woh, qnsT)                   # (batch_size, context_len, question_len)
                               
            S = tf.expand_dims(wh, 2) + tf.expand_dims(wq, 1) + whoq  # (batch_size, context_len, question_len)
            
            #c2q
            attn_mask_qs = tf.expand_dims(qns_mask, 1)         # (batch_size, 1, question_len)
            _, attn_dist = masked_softmax(S, attn_mask_qs, 2)  # (batch_size, context_len, question_len)
            Ut = tf.matmul(attn_dist, qns)                     # (batch_size, context_len, hidden_size*2)
            Ut = tf.transpose(Ut, perm=[0, 2, 1])              # (batch_size, hidden_size*2, context_len)
                              
            #q2c
            Smaxcol = tf.reduce_max(S, axis=2)                 # (batch_size, context_len)
#             attn_mask_co = tf.expand_dims(context_mask, 2)       # (batch_size, context_len, 1)
            _, b = masked_softmax(Smaxcol, context_mask, 1)    # (batch_size, context_len)
            b = tf.expand_dims(b, axis=1)                      # (batch_size, 1, context_len)
            ht = tf.matmul(b, context)                         # (batch_size, 1, hidden_size*2)
            ht = tf.squeeze(ht)                                # (batch_size, hidden_size*2)
            
            batch_size = tf.shape(ht)[0]
            dim1_size  = tf.shape(ht)[1]
            dim2_size  = tf.shape(Ut)[2]
            ht = tf.reshape(ht, (-1,))
            Ht = tf.reshape(tf.tile(ht, [dim2_size]), (batch_size, dim1_size, dim2_size))
                              # (batch_size, hidden_size*2, context_len)
                              
            H = tf.transpose(context, perm=[0, 2, 1])          # (batch_size, hidden_size*2, context_len)
            HoUt = tf.multiply(H, Ut)
            HoHt = tf.multiply(H, Ht)
                              
            # output
            G = tf.concat([H, Ut, HoUt, HoHt], axis=1)         # (batch_size, hidden_size*8, context_len)
            output = tf.transpose(G, perm=[0,2,1])
            output = tf.nn.dropout(output, self.keep_prob)
                              
            return attn_dist, b, output
class Bidaf2(object):
    def __init__(self, keep_prob, context_vec_size, qn_vec_size):
        self.keep_prob = keep_prob
        self.context_vec_size = context_vec_size
        self.qn_vec_size = qn_vec_size

    def build_graph(self, context, context_mask, qns, qns_mask, scope):
        with vs.variable_scope(scope):

            # context:  (batch_size, context_len, hidden_size*2)
            # qns    :  (batch_size, question_len, hidden_size*2)
            for t in range(4):
                Wq=tf.get_variable("Wq"+str(t),shape=[self.qn_vec_size/4,self.qn_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                Wc=tf.get_variable("Wc"+str(t),shape=[self.context_vec_size/4,self.context_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                qns_proj=tf.transpose(tf.tensordot(Wq,qns,axes=((1,),(2,))),perm=[1,2,0])
                context_proj=tf.transpose(tf.tensordot(Wq,context,axes=((1,),(2,))),perm=[1,2,0])            
                # assert self.context_vec_size == self.qn_vec_size
                ws1 = tf.get_variable("ws1"+str(t),shape=[self.qn_vec_size/4,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                ws2 = tf.get_variable("ws2"+str(t),shape=[self.qn_vec_size/4,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                ws3 = tf.get_variable("ws3"+str(t),shape=[self.qn_vec_size/4,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                
                wh = tf.tensordot(context_proj, ws1, [[2], [0]])   # (batch_size, context_len)
                wq = tf.tensordot(qns_proj, ws2, [[2], [0]])       # (batch_size, question_len)
                woh = tf.multiply(context_proj, ws3)               # (batch_size, context_len, hidden_size*2)
                qnsT = tf.transpose(qns_proj, perm=[0, 2, 1])      # (batch_size, hidden_size*2, question_len)
                whoq = tf.matmul(woh, qnsT)                   # (batch_size, context_len, question_len)
                                   
                S = tf.expand_dims(wh, 2) + tf.expand_dims(wq, 1) + whoq  # (batch_size, context_len, question_len)
                #qns_t = tf.transpose(qns_proj, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
                #S = tf.matmul(context_proj, qns_t)               
                #c2q
                attn_mask_qs = tf.expand_dims(qns_mask, 1)         # (batch_size, 1, question_len)
                _, attn_dist = masked_softmax(S, attn_mask_qs, 2)  # (batch_size, context_len, question_len)
                Ut = tf.matmul(attn_dist, qns)                     # (batch_size, context_len, hidden_size*2)
                Ut = tf.transpose(Ut, perm=[0, 2, 1])              # (batch_size, hidden_size*2, context_len)
                                  
                #q2c
                Smaxcol = tf.reduce_max(S, axis=2)                 # (batch_size, context_len)
    #             attn_mask_co = tf.expand_dims(context_mask, 2)       # (batch_size, context_len, 1)
                _, b = masked_softmax(Smaxcol, context_mask, 1)    # (batch_size, context_len)
                b = tf.expand_dims(b, axis=1)                      # (batch_size, 1, context_len)
                ht = tf.matmul(b, context)                         # (batch_size, 1, hidden_size*2)
                ht = tf.squeeze(ht)                                # (batch_size, hidden_size*2)
                
                batch_size = tf.shape(ht)[0]
                dim1_size  = tf.shape(ht)[1]
                dim2_size  = tf.shape(Ut)[2]
                ht = tf.reshape(ht, (-1,))
                Ht = tf.reshape(tf.tile(ht, [dim2_size]), (batch_size, dim1_size, dim2_size))
                                  # (batch_size, hidden_size*2, context_len)
                                  
                H = tf.transpose(context, perm=[0, 2, 1])          # (batch_size, hidden_size*2, context_len)
                HoUt = tf.multiply(H, Ut)
                HoHt = tf.multiply(H, Ht)
                                  
                # output
                G = tf.concat([H, Ut,HoUt, HoHt], axis=1)         # (batch_size, hidden_size*6, context_len)
                G = tf.transpose(G, perm=[0,2,1])
                if t==0:
                    output = G 
                else:
                    output=tf.concat([output,G],axis=2)
            output = tf.nn.dropout(output, self.keep_prob)
                              
            return attn_dist, b, output                              
class ComplexAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys,scope):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope(scope):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            for t in range(4):
	            Wv_sim=tf.get_variable("Wv_sim"+str(t),shape=[self.key_vec_size/4,self.value_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
	            values_t_proj=tf.transpose(tf.tensordot(Wv_sim, values_t,axes=((1,),(1,))),perm=[1,0,2]) # shape (batch_size, num_keys, num_values)
	            Wk_sim=tf.get_variable("Wk_sim"+str(t),shape=[self.value_vec_size/4,self.key_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
	            keys_proj=tf.transpose(tf.tensordot(Wk_sim,keys,axes=((1,),(2,))),perm=[1,2,0])
	            attn_logits = tf.matmul(keys_proj, values_t_proj) # shape (batch_size, num_keys, num_values)
	            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
	            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values
	            # Use attention distribution to take weighted sum of values
	            values_t_proj=tf.transpose(values_t_proj,perm=[0,2,1])
	            if t==0:
	            	output = tf.matmul(attn_dist, values_t_proj) # shape (batch_size, num_keys, value_vec_size)
	            else:
	            	output=tf.concat([output,tf.matmul(attn_dist, values_t_proj)],axis=2)
            # output1 = tf.matmul(attn_dist, keys) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, scope):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"+scope):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output
class SelfAttn(object):


    def __init__(self, keep_prob, vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.vec_size = vec_size

    def build_graph(self, values, values_mask, scope):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope(scope):
            Wv=tf.get_variable("Wv",shape=[self.vec_size,self.vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())      
            Wk=tf.get_variable("Wk",shape=[self.vec_size,self.vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())      
            v=tf.get_variable("v",shape=[self.vec_size,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            keys_proj=tf.transpose(tf.tensordot(Wv,values,axes=((1,),(2,))),perm=[1,2,0])
            attn_logits=[]
            loop=tf.unstack(values,axis=1)
            for i in loop:
                attn_logits.append(tf.reduce_sum(v*tf.nn.tanh(keys_proj+tf.expand_dims(tf.matmul(i,Wk),1)),axis=2))

            attn_logits=tf.stack(attn_logits)
            attn_logits=tf.transpose(attn_logits,perm=[1,0,2])
            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            #attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output
class DotProductAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self,queries,keys,values,mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("DotProductAttn"):

            # Calculate attention distribution
            for t in range(4):
                Wq=tf.get_variable("Wq"+str(t),shape=[self.key_vec_size/4,self.key_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                Wk=tf.get_variable("Wk"+str(t),shape=[self.key_vec_size/4,self.key_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                Wv=tf.get_variable("Wv"+str(t),shape=[self.value_vec_size/4,self.value_vec_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                keys_proj=tf.transpose(tf.tensordot(Wk,keys,axes=((1,),(2,))),perm=[1,2,0])
                queries_proj=tf.transpose(tf.tensordot(Wq,queries,axes=((1,),(2,))),perm=[1,2,0])
                values_proj=tf.transpose(tf.tensordot(Wv,values,axes=((1,),(2,))),perm=[1,2,0])
                keys_t = tf.transpose(keys_proj, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
                attn_logits = tf.matmul(queries_proj, keys_t)/tf.sqrt(tf.to_float(queries_proj.shape[2])) # shape (batch_size, num_keys, num_values)
                attn_logits_mask = tf.expand_dims(mask, 1)
                _,attn_dist=masked_softmax(attn_logits, attn_logits_mask, 2)
	            # Use attention distribution to take weighted sum of values
               	if t==0:
                    output = tf.matmul(attn_dist, values_proj) # shape (batch_size, num_keys, value_vec_size)
                else:
                    output=tf.concat([output,tf.matmul(attn_dist, values_proj)],axis=2)
            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output
def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
