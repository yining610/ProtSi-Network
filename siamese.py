import tensorflow as tf

class Siamese(tf.keras.layers.Layer):
    """
    Implementation of Siamese network
    """
    def __init__(self, max_length, num_classes, shots=1):
        super(Siamese, self).__init__()
        self.max_length = max_length
        self.num_classes = num_classes
        self.shots = shots # default is number of query_set shots

        # can be further modified. 
        
        # for encoding student answer
        self.encoder1 = tf.keras.Sequential([
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu')
        ]
        )
        
#         # for encoding model answer
#         self.encoder2 = tf.keras.Sequential([
#             tf.keras.layers.LSTM(64)
#         ]
#         )
        
    def call(self, student_answer_emb, model_answer_emb):
        '''
        using pre-trained bert model to create embedding layer for either support set or query set
        
        student_answer_emb: [shots, max_length, embedding_length]
        model_answer_emb: [1, max_length, embedding_length]
        '''
        embedding_size = 768
        student_answer_emb = tf.reshape(student_answer_emb, [self.num_classes*self.shots, self.max_length, embedding_size])
    
        # shape : [shots, 64]
        student_answer = self.encoder1(student_answer_emb)
        # shape: [1, 64]
        model_answer = self.encoder1(model_answer_emb)

        # shape: [shots, 64]
        # could further modified
        distance = tf.square(model_answer - student_answer)

        distance = tf.reshape(distance, shape=[self.num_classes, self.shots, 32])

        return distance
        