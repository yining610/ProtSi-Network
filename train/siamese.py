import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from config import *

class Siamese(tf.keras.layers.Layer):
    """
    Implementation of Siamese network
    """
    def __init__(self, max_length, embedding_size):
        super(Siamese, self).__init__()
        self.max_length = max_length
        self.embedding_size = embedding_size  # d_model is the embedding_size. refers to dataset.py\

        self.encoder1 = tf.keras.Sequential([
            tf.keras.layers.Masking(mask_value=0.),
            # tf.keras.layers.LSTM(384, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            # tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(64, dropout=0.2)
        ])

        self.dense_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization()
        ])
        
        self.dense_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization()
        ])
        
        self.dense_3 = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization()
        ])

        # Euclidean distance
        self.distance_function = lambda x,y: tf.square(x - y)
        # Pearson distance
        # self.distance_function = lambda x,y: tfp.stats.correlation(x, y)
        
    def call(self, support_student_answer_emb, query_student_answer_emb, unlabel_answer_emb, model_answer_emb):
        '''
        Encoding the answers in support and query set and model answer
        Compare model answer with either support answer or query answer
        
        Input: 
            support_student_answer_emb: [batch_size, num_classes, support_shots, max_tokens, embedding_length]
            query_student_answer_emb : [batch_size, num_classes, query_shots, max_tokens, embedding_length]
            model_answer_emb: [batch_size, max_tokens, embedding_length]
            unlabel_answer_emb: [batch_size, num_unlabel_samples, (num_return_sequences+1), max_tokens, embedding_length]

        Output:
            distance_support: [batch_size*num_classes*support_shots, encoding_size]
            distance_query: [batch_size*num_classes*query_shots, encoding_size]
            distance_unlabel: [batch_size*num_unlabel_samples*(num_return_sequences+1), encoding_size]
        '''

        num_classes = support_student_answer_emb.shape[1]
        support_shots = support_student_answer_emb.shape[2]
        query_shots = query_student_answer_emb.shape[2]
        num_unlabel_samples = unlabel_answer_emb.shape[1]
        num_sequences = unlabel_answer_emb.shape[2]        

        support_student_answer_emb = tf.reshape(support_student_answer_emb, [-1, self.max_length, self.embedding_size])
        query_student_answer_emb = tf.reshape(query_student_answer_emb, [-1, self.max_length, self.embedding_size])
        unlabel_answer_emb = tf.reshape(unlabel_answer_emb, [-1, self.max_length, self.embedding_size])
        
        # shape : [batch_size*num_classes*support_shots, 64]
        support_student_answer = self.encoder1(support_student_answer_emb)
        # shape : [batch_size*num_classes*query_shots, 64]
        query_student_answer = self.encoder1(query_student_answer_emb)
        # shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), 64]
        unlabel_answer = self.encoder1(unlabel_answer_emb)
        model_answer = self.encoder1(model_answer_emb)

        # shape : [batch_size*num_classes*support_shots, 64]
        model_answer_support = tf.repeat(model_answer,repeats=num_classes*support_shots,axis=0)
        # shape : [batch_size*num_classes*query_shots, 64]
        model_answer_query = tf.repeat(model_answer,repeats=num_classes*query_shots,axis=0)
        # shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), 64]
        model_answer_unlabel = tf.repeat(model_answer,repeats=num_unlabel_samples*num_sequences,axis=0)

        # shape : [batch_size*num_classes*support_shots, 32]
        distance_support = self.dense_1(tf.concat([support_student_answer, model_answer_support], axis=1))
        #distance_support = self.dense_1(tf.math.subtract(support_student_answer, model_answer_support))
        # shape : [batch_size*num_classes*query_shots, 32]
        distance_query = self.dense_2(tf.concat([query_student_answer, model_answer_query], axis=1))
        # distance_query = self.dense_2(tf.math.subtract(query_student_answer, model_answer_query))
        # shape: [batch_size*num_unlabel_samples*(num_return_sequences+1), 32]
        distance_unlabel = self.dense_3(tf.concat([unlabel_answer, model_answer_unlabel], axis=1))
        # distance_unlabel = self.dense_3(tf.math.subtract(unlabel_answer, model_answer_unlabel))

        # distance_support = self.distance_function(support_student_answer, model_answer)
        # distance_query = self.distance_function(query_student_answer, model_answer)
        # distance_unlabel = self.distance_function(unlabel_answer, model_answer)

        return distance_support, distance_query, distance_unlabel
        