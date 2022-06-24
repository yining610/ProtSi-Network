import tensorflow as tf

def calc_euclidian_dists(x, y):
    """
    Calculate euclidian distance between two 3D tensors.
    Args:
        x (tf.Tensor): query
        y (tf.Tensor): prototypes_label
    Returns (tf.Tensor): 2-dim tensor with distances.
    """    
    n = x.shape[1]
    m = y.shape[1]
    x = tf.tile(tf.expand_dims(x, 2), [1, 1, m, 1])
    y = tf.tile(tf.expand_dims(y, 1), [1, n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 3)

class Prototypical(tf.keras.layers.Layer):
    def __init__(self, 
                 batch_size, 
                 average,
                 num_classes,
                 n_support,
                 n_query,
                 encoding_size,
                 num_unlabel_samples,
                 num_return_sequences,
                ):
        super(Prototypical, self).__init__()
        self.batch_size = batch_size
        self.average = average
        self.num_unlabel_samples = num_unlabel_samples
        self.num_return_sequences = num_return_sequences
        self.num_classes = num_classes
        self.n_query = n_query
        self.n_support = n_support
        self.encoding_size = encoding_size

    def call(self, support, query, unlabel):
        '''
        Inputs:
            support: [batch_size*num_classes*support_shots, 32]
            query: [batch_size*num_classes*query_shots, 32]
            unlabel: [batch_size*num_unlabel_samples*(num_return_sequences+1), 32]

        Outputs:
            log_p_y, unsupervised_dist
        '''
        support = tf.reshape(support, [self.batch_size, self.num_classes, self.n_support, self.encoding_size])
        query = tf.reshape(query, [self.batch_size, self.num_classes, self.n_query, self.encoding_size])
        unlabel = tf.reshape(unlabel, [self.batch_size, self.num_unlabel_samples, (self.num_return_sequences+1), self.encoding_size])

        # split unlabel answer to original answers and paraphrased answers
        # [batch_size, num_unlabel_samples, 1, encoding_size]
        random_answer = tf.slice(unlabel, [0,0,0,0], [self.batch_size, self.num_unlabel_samples, 1, self.encoding_size])
        # [batfch_size, num_unlabel_samples, num_return_sequencecs, encoding_size] 
        paraphrase_answer = tf.slice(unlabel, [0,0,1,0],[self.batch_size, self.num_unlabel_samples, self.num_return_sequences, self.encoding_size])
        
        # [batch_sizze, num_classes, 32]
        prototypes_label = tf.math.reduce_mean(support, axis=2)
        # [batch_size, num_unlabel_samples, encoding_size]
        prototypes_unlabel = tf.math.reduce_mean(paraphrase_answer, axis=2)

        if self.average: 
            # average all prototypes_label over batche
            prototypes_label = tf.math.reduce_mean(prototypes_label, axis=0) 
            prototypes_label = tf.reshape(prototypes_label, shape=[1, self.num_classes, self.encoding_size])

        query = tf.reshape(query,[self.batch_size, self.num_classes*self.n_query, self.encoding_size])
        random_answer = tf.reshape(random_answer, [self.batch_size, self.num_unlabel_samples*1, self.encoding_size])

        # shape = [batch_size, num_classes*query_shots, num_classes]
        dists_label = calc_euclidian_dists(query, prototypes_label)
        # shape = [batch_size, num_unlabel_samples, num_unlabel_samples]
        dists_unlabel = calc_euclidian_dists(random_answer, prototypes_unlabel)

        return dists_label, dists_unlabel, unlabel
    