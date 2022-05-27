import tensorflow as tf
import numpy as np

class Inference(tf.keras.layers.Layer):
    '''
    Implement Inference Network from paper: Meta Learning from Tasks with Heterogeneous Attribute Spaces
    using Tensorflow
    '''
    def __init__(self,
                 embedding_size,
                 encoding_method,
                 num_classes,
                 support_shots,
                 query_shots,
                 batch_size,
                 num_unlabel_samples,
                 num_return_sequences):

        super(Inference, self).__init__()    
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.support_shots = support_shots
        self.query_shots = query_shots
        self.batch_size = batch_size
        self.num_unlabel_samples = num_unlabel_samples

        self.new_return_sequences = num_return_sequences + 1  

        if encoding_method == 1:
            self.encoding_size = 64  # refer to siamese.py file. encoding_size is the number of attributes
        else:
            self.encoding_size = 90 * 768 

        self.fvc_bar =  tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.gvc_bar = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.fu = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.gu = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.fvc = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.gvc = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.fz = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

        self.gz = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu')
            # tf.keras.layers.Dense(32, activation='relu'),
            # tf.keras.layers.Dense(32, activation='relu')
        ])

    # Inference Network
    def call(self, support_distance, support_label, unlabel_distance=None, query_distance=None):
        '''
        new_return_sequence (int) = num_return_sequence + 1
        the total unlabel answers include new paraphrased answer (num_return_sequence) and origianl randomly sampled answer (1)

        Input:
            support_distance: [batch_size*num_classes*support_shots, encoding_size]
            query_distance: [batch_size*num_classes*query_shots, encoding_size]
            unlabel_distance: [batch_size*num_unlabel_samples*new_return_sequences, encoding_size]
            support_label: [batch_size, 15]

        Output;
            new_distance: [batch_size, num_classes(num_unlabel_samples), num_shots(new_return_sequences), 32]
        '''

        distance_1 = tf.reshape(support_distance, shape=[self.batch_size*self.num_classes*self.support_shots, self.encoding_size, 1])
        v_bar = self.gvc_bar(tf.reduce_mean(self.fvc_bar(distance_1), axis=0))
        tasks_label_1 = tf.reshape(
                                   tf.one_hot(
                                              tf.cast(
                                                     tf.reshape(support_label, 
                                                                shape=[self.batch_size*self.num_classes*self.support_shots, 1]), 
                                              tf.int32), 
                                    self.num_classes),
                                    shape=[self.batch_size*self.num_classes*self.support_shots, self.num_classes, 1]
                                  )
        c_bar = self.gvc_bar(tf.reduce_mean(self.fvc_bar(tasks_label_1), axis=0))
        # concatenate vi_bar and x_ni
        concate1 = tf.concat([tf.repeat(tf.expand_dims(v_bar,axis=0), self.batch_size*self.num_classes*self.support_shots, axis=0), distance_1], axis=2)

        # encode over all the attributes
        # shape: [batch_size*num_classes * shots, 32]
        alpha = tf.reduce_mean(self.fu(concate1), axis=1)

        # concatenate cj_bar and y_ni
        concate2 = tf.concat([tf.repeat(tf.expand_dims(c_bar,axis=0), self.batch_size*self.num_classes*self.support_shots, axis=0), tf.cast(tasks_label_1, dtype=tf.float32)], axis=2)

        # encode over all the responese of the instances
        # shape: [batch_size*num_classes * shots, 32]
        beta = tf.reduce_mean(self.fu(concate2), axis=1)

        # shape: [batch_szie * num_classes * shots, 32]
        u_n = self.gu(alpha + beta)

        # concatenate u_n and x_ni
        concate3 = tf.concat([tf.repeat(tf.expand_dims(u_n,axis=1), self.encoding_size, axis=1), distance_1], axis=2)

        # encode the relationship of the attribute to the others
        # shape: [encoding_size, 32]
        v = self.gvc(tf.reduce_mean(self.fvc(concate3), axis=0))

        # concate latent attribute verctors with query or support's distance
        if tf.is_tensor(query_distance):
            distance_2 = tf.reshape(query_distance, shape=[self.batch_size*self.num_classes*self.query_shots, self.encoding_size, 1])
            # shape: [batch_size*num_classes*query_shots, encoding_size, 33]
            concate4 = tf.concat([tf.repeat(tf.expand_dims(v, axis=0), self.batch_size*self.num_classes*self.query_shots, axis=0), distance_2], axis=2)
            # shape: [batch_size*num_classes*query_shots, 32]
            new_distance = self.gz(tf.reduce_mean(self.fz(concate4), axis=1))
            new_distance = tf.reshape(new_distance, shape=[self.batch_size, self.num_classes, self.query_shots, 32])
            return new_distance
        elif tf.is_tensor(unlabel_distance):
            distance_2 = tf.reshape(unlabel_distance, shape=[self.batch_size*self.num_unlabel_samples*self.new_return_sequences, self.encoding_size, 1])
            concate4 = tf.concat([tf.repeat(tf.expand_dims(v, axis=0), self.batch_size*self.num_unlabel_samples*self.new_return_sequences, axis=0), distance_2], axis=2)
            new_distance = self.gz(tf.reduce_mean(self.fz(concate4), axis=1))
            new_distance = tf.reshape(new_distance, shape=[self.batch_size, self.num_unlabel_samples, self.new_return_sequences, 32])
            return new_distance
        else:
            # shape: [batch_size*num_classes*shots, encoding_size, 33]
            concate4 = tf.concat([tf.repeat(tf.expand_dims(v, axis=0), self.batch_size*self.num_classes*self.support_shots, axis=0), distance_1], axis=2)
            # shape: [batch_size*num_classes*shots, 32]
            new_distance = self.gz(tf.reduce_mean(self.fz(concate4), axis=1))
            new_distance = tf.reshape(new_distance, shape=[self.batch_size, self.num_classes, self.support_shots, 32])
            return new_distance