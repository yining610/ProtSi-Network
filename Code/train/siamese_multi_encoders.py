import tensorflow as tf
import numpy as np
from transformer import Encoder, create_padding_mask

class Siamese(tf.keras.layers.Layer):
    """
    Implementation of Siamese network
    num_layers: number of blocks in transformer encoder part
    """
    def __init__(self, max_length, encoding_method, embedding_size):
        super(Siamese, self).__init__()
        self.max_length = max_length
        self.method = encoding_method         # encoding method
        self.embedding_size = embedding_size  # d_model is the embedding_size. refers to dataset.py

        if encoding_method == 1:
            # for encoding student answer
            self.encoder1 = tf.keras.Sequential([
                tf.keras.layers.Masking(mask_value=0.),
                tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
                tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.1),
                tf.keras.layers.LSTM(64, dropout=0.1)
            ]
            )

            self.encoder2 = tf.keras.Sequential([
                tf.keras.layers.Masking(mask_value=0.),
                tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
                tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.1),
                tf.keras.layers.LSTM(64, dropout=0.1)
            ]
            )
        elif encoding_method == 2:
            # block数量需要小，因为数据体量较小
            num_blocks = 1
            dff = 2048
            num_heads = 8
            dropout_rate = 0.1
            max_tokens = max_length # refer to dataset.py file

            self.transformer_encoder = tf.keras.Sequential([
                Encoder(num_layers=num_blocks, d_model=self.embedding_size, num_heads=num_heads,
                        dff=dff, max_tokens = max_tokens, rate=dropout_rate)  # diff is optimizable
            ])
        
    def call(self, support_student_answer_emb, query_student_answer_emb, unlabel_answer_emb, model_answer_emb):
        '''
        using pre-trained bert model to create embedding layer for either support set or query set
        
        Input: 
            support_student_answer_emb: [batch_size, num_classes, support_shots, max_tokens, embedding_length]
            query_student_answer_emb : [batch_size, num_classes, query_shots, max_tokens, embedding_length]
            model_answer_emb: [1, max_tokens, embedding_length]
            unlabel_answer_emb: [batch_size, num_unlabel_samples, (num_return_sequences+1), max_tokens, embedding_length]

        Output:
            distance_support: [batch_size*num_classes*support_shots, encoding_size]
            distance_query: [batch_size*num_classes*query_shots, encoding_size]
            distance_unlabel: [batch_size*num_unlabel_samples*(num_return_sequences+1), encoding_size]
        '''
        # method 1 use normal RNN encoder
        if self.method == 1:             
            support_student_answer_emb = tf.reshape(support_student_answer_emb, [-1, self.max_length, self.embedding_size])
            query_student_answer_emb = tf.reshape(query_student_answer_emb, [-1, self.max_length, self.embedding_size])
            unlabel_answer_emb = tf.reshape(unlabel_answer_emb, [-1, self.max_length, self.embedding_size])
            
            # shape : [batch_size*num_classes*support_shots, 64]
            support_student_answer = self.encoder1(support_student_answer_emb)

            # shape : [batch_size*num_classes*query_shots, 64]
            query_student_answer = self.encoder2(query_student_answer_emb)

            # shape: [1, 64]
            model_answer_support = self.encoder1(model_answer_emb)
            model_answer_query = self.encoder2(model_answer_emb)
            model_answer_unlabel = self.encoder1(model_answer_emb)
            
            # shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), 64]
            unlabel_answer = self.encoder1(unlabel_answer_emb)
                    
            # shape: [batch_size*num_classes*shots, encoding_size]
            distance_support = tf.square(support_student_answer - model_answer_support)
            distance_query = tf.square(query_student_answer - model_answer_query)

            # shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), encoding_size]
            # further modified
            distance_unlabel = tf.square(unlabel_answer - model_answer_unlabel)

        # method 2 use Transformer Encoder
        elif self.method == 2: 
            num_classes = support_student_answer_emb.shape[1]
            support_shots = support_student_answer_emb.shape[2]
            query_shots = query_student_answer_emb.shape[2]
            num_unlabel_samples = unlabel_answer_emb.shape[1]
            num_return_sequences = unlabel_answer_emb.shape[2]

            support_student_answer_emb = tf.reshape(support_student_answer_emb, [-1, self.max_length, self.embedding_size])
            query_student_answer_emb = tf.reshape(query_student_answer_emb, [-1, self.max_length, self.embedding_size])
            unlabel_answer_emb = tf.reshape(unlabel_answer_emb, [-1, self.max_length, self.embedding_size])

            padding_mask1 = create_padding_mask(support_student_answer_emb)
            # output shape: [batch_size*num_classes*support_shots, max_tokens, embedding_size]
            support_student_answer = self.transformer_encoder(support_student_answer_emb, training=True, mask=padding_mask1)
            # output shape: [batch_size*num_classes*support_shots, max_tokens*embedding_size]
            support_student_answer = tf.reshape(support_student_answer, shape=[-1, self.max_length*self.embedding_size])

            padding_mask2 = create_padding_mask(query_student_answer_emb)
            query_student_answer = self.transformer_encoder(query_student_answer_emb, training=True, mask=padding_mask2)
            # output shape: [batch_size*num_classes*query_shots, max_tokens*embedding_size]
            query_student_answer = tf.reshape(query_student_answer, shape=[-1, self.max_length*self.embedding_size])

            padding_mask3 = create_padding_mask(unlabel_answer_emb)
            unlabel_answer = self.transformer_encoder(unlabel_answer_emb, training=True, mask=padding_mask3)
                        # output shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), max_tokens*embedding_size]
            unlabel_answer = tf.reshape(unlabel_answer, shape=[-1, self.max_length*self.embedding_size])

            padding_mask4 = create_padding_mask(model_answer_emb)
            model_answer = self.transformer_encoder(model_answer_emb, training=True, mask=padding_mask4)
            # output shape: [1, max_tokens*embedding_size]
            model_answer = tf.reshape(model_answer, shape=[1, self.max_length*self.embedding_size])

            # shape: [batch_size*num_classes*shots, max_length*embedding_size]
            distance_support = tf.square(support_student_answer - model_answer)
            distance_query = tf.square(query_student_answer - model_answer)

            # shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), max_length*embedding_size]
            # further modified
            distance_unlabel = tf.square(unlabel_answer - model_answer)

        # use consine similarity. If you need to use consine similairity as similarity function, try to simplifiy encoder network. 
        # use consine similarity will return a numerical scalar in one dimension, so lots of information of prototye will be lost
       
        # distance = tf.reshape(tf.matmul(student_answer,tf.transpose(model_answer)), shape=[-1]) / (tf.norm(student_answer,axis=1) * tf.norm(model_answer))
        # distance = tf.reshape(distance, shape=[self.num_classes, self.shots, 1])

        return distance_support, distance_query, distance_unlabel
        