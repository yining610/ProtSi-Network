import tensorflow as tf

def read_dataset(example):
    # Attention, here we explicitly define the shape of labels
    feature_map = {'tasks': tf.io.FixedLenFeature((), tf.string),
                   'tasks_query': tf.io.FixedLenFeature((), tf.string),
                   'tasks_label': tf.io.FixedLenFeature((20), tf.float32),  # 3 * 5 / 4 * 5
                   'query_labels': tf.io.FixedLenFeature((4), tf.float32),  # 3 * 2 / 4 * 2
                   'paraphrase_data': tf.io.FixedLenFeature((), tf.string),
                   'training_classes' : tf.io.FixedLenFeature((), tf.int64, default_value=None),
                   'training_support_shots' : tf.io.FixedLenFeature((), tf.int64, default_value=None),
                   'training_query_shots' : tf.io.FixedLenFeature((), tf.int64, default_value=None),
                   'max_length' : tf.io.FixedLenFeature((), tf.int64, default_value=None),
                   'embedding_size' : tf.io.FixedLenFeature((), tf.int64, default_value=None),
                   'num_unlabel_samples' : tf.io.FixedLenFeature((), tf.int64, default_value=None),
                   'num_return_sequences' : tf.io.FixedLenFeature((), tf.int64, default_value=None)                  
                   }
    parsed_example = tf.io.parse_single_example(example, features=feature_map)

    tasks = tf.io.decode_raw(parsed_example['tasks'], out_type=tf.float32)
    tasks_query = tf.io.decode_raw(parsed_example['tasks_query'], out_type=tf.float32)
    paraphrase_data = tf.io.decode_raw(parsed_example['paraphrase_data'], out_type=tf.float32)

    tasks_label = parsed_example['tasks_label']
    query_labels = parsed_example['query_labels']

    # load parameters
    training_classes = parsed_example["training_classes"]
    training_support_shots = parsed_example["training_support_shots"]
    training_query_shots = parsed_example["training_query_shots"]
    max_length = parsed_example["max_length"]
    embedding_size = parsed_example["embedding_size"]
    num_unlabel_samples = parsed_example["num_unlabel_samples"]
    num_return_sequences = parsed_example["num_return_sequences"]    
    
    tasks = tf.reshape(tasks, [training_classes, training_support_shots, max_length, embedding_size])
    tasks_query = tf.reshape(tasks_query, [training_classes, training_query_shots, max_length, embedding_size])
    paraphrase_data = tf.reshape(paraphrase_data, [num_unlabel_samples, (num_return_sequences+1), max_length, embedding_size])
    return (tasks, tasks_query, paraphrase_data, tasks_label), (query_labels)

# Read model answer
def read_model(example):
    feature_map = {'model_answer': tf.io.FixedLenFeature((), tf.string),
                   'max_length': tf.io.FixedLenFeature((), tf.int64),
                   'embedding_size': tf.io.FixedLenFeature((), tf.int64)
                  }
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    model_answer = tf.io.decode_raw(parsed_example["model_answer"], out_type=tf.float32)
    max_length = parsed_example['max_length']
    embedding_size = parsed_example['embedding_size']
    
    model_answer = tf.reshape(model_answer, [1, max_length, embedding_size])
    return model_answer