import os
import time
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from dataset import Dataset
from siamese import Siamese
from prototypical import Prototypical
from transformers import AutoTokenizer,TFAutoModel

train_dataset = Dataset(question_set=1, training=True)
test_dataset = Dataset(question_set=1, training=False)
# set hyperparameters
solution_sentences_1= 'how much vinegar was used in each container. what type of vinegar was used in each containe. what materials to test. what size or surface area of materials should be used. how long each sample was rinsed in distilled water. what drying method to use. what size or type of container to use'
model_answer = solution_sentences_1
num_classes = 4 # 如果不是4， prototypical network里有一点bug
support_shots = 3 # support_shots
max_length = train_dataset.max_length
epochs = 20
episodes = 100
embedding_size = 768

path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(path)
bert = TFAutoModel.from_pretrained(path)

model_tokens =  tokenizer(model_answer, return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)
model_answer = bert(model_tokens)[0]

# prepare dataset
tasks = []
tasks_query = []
query_labels = []
for episode in range(episodes):
    new_task, _ , new_query , new_label = train_dataset.get_mini_dataset(support_shots, num_classes, split=True)
    # Tokenization
    new_task_tokens = tokenizer(list(new_task.flat), return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)
    # Embedding
    new_task_emb = bert(new_task_tokens)[0]
    tasks.append(new_task_emb)
    
    new_query_tokens = tokenizer(list(new_query.flat), return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)
    new_query_emb = bert(new_query_tokens)[0]
    tasks_query.append(new_query_emb)
    
    query_labels.append(new_label)

tasks = tf.stack(tasks, axis=0)
tasks_query= tf.stack(tasks_query, axis=0)
query_labels = np.stack(query_labels,axis=0)

loss_tracker = tf.keras.metrics.Mean('loss')
accuracy_tracker = tf.keras.metrics.Mean('acc')
mae_metric = tf.keras.metrics.MeanAbsoluteError(name = 'mae')

class ProtSi(tf.keras.Model):    
    def train_step(self, data):
        # x is the inputs: [support_set, query_set] , y is the query label
        inputs, y = data  
        with tf.GradientTape() as tape:
            log_p_y = self(inputs)
            n_query = 1
            n_class = 4
            y_hat = tf.cast(tf.reshape(y, [n_class,n_query]), tf.int32)
            y_onehot = tf.cast(tf.one_hot(y_hat, n_class), tf.float32)
            # use log softmax to compute loss
            loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        
            y_pred = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)

            eq = tf.cast(tf.equal(y_pred, 
                                  tf.cast(tf.reshape(y,[n_class,n_query]), tf.int32)), tf.float32)
            acc = tf.reduce_mean(eq)
        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # compute our own metrics
        loss_tracker.update_state(loss)
        accuracy_tracker.update_state(acc)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result(), 'acc': accuracy_tracker.result()}
        
    @property
    def metrics(self):
        return [loss_tracker, mae_metric, accuracy_tracker]

# siamese network only share BERT embedding weights
siamese_network1 = Siamese(max_length,num_classes, shots = support_shots) # between support set and model answer
siamese_network2 = Siamese(max_length,num_classes, shots = 1) # between query set and model answer
prototypical_network = Prototypical(num_classes)

inputs1 = tf.keras.Input(shape=(num_classes * support_shots, max_length, embedding_size))
inputs2 = tf.keras.Input(shape=(num_classes * 1, max_length, embedding_size))
# shape: [num_classes, support_shots, embedding_size]
support_dis = siamese_network1(inputs1, model_answer) # distance between support set and model answer
# shape: [num_classes, query_shots, embedding_size]
query_dis = siamese_network2(inputs2, model_answer) # distance between query set and model answer

outputs = prototypical_network(support_dis, query_dis)
model = ProtSi([inputs1, inputs2], outputs)

model.compile(optimizer='adam')
  
model.summary()

# one task -> batch size = 1
model.fit([tasks, tasks_query], query_labels, epochs=epochs, batch_size=1)

