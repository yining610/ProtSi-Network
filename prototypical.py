import tensorflow as tf

def calc_euclidian_dists(x, y):
    """
    Calculate euclidian distance between two 3D tensors.
    Args:
        x (tf.Tensor):
        y (tf.Tensor):
    Returns (tf.Tensor): 2-dim tensor with distances.
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)

class Prototypical(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        
        super(Prototypical, self).__init__()
        self.num_classes = num_classes
    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1] # shots for support set
        n_query = query.shape[1] # shots for query set. Always be 1
        embedding_size = support.shape[2]
        
        # y = query_label.reshape(n_class,n_query).astype(int)
        # y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)
        
        prototypes = tf.math.reduce_mean(support, axis=1)
        query = tf.reshape(query,[self.num_classes,embedding_size])
        
        dists = calc_euclidian_dists(query, prototypes)
        
        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [self.num_classes, n_query, -1])
        
        return log_p_y
    
#         loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        
#         eq = tf.cast(tf.equal(
#         tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32), 
#         tf.cast(y, tf.int32)), tf.float32)
#         acc = tf.reduce_mean(eq)
        
        # return loss, acc