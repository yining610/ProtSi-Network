import sys
import os
import time
import numpy as np
import tensorflow as tf
# tf.compat.v1.executing_eagerly()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# from siamese_multi_encoders import Siamese
from siamese import Siamese
from prototypical import Prototypical
from inference import Inference
from decoder import read_dataset, read_model
import datetime

print(f'Current working directory: {os.getcwd()}')

# set parameters
from config import *

train_path = os.path.join('/root/autodl-tmp/Code/data/Q5_5_1_4_5_5_15_0.5_5_90_2400', "train_dataset.tfrecords")
val_path = os.path.join('/root/autodl-tmp/Code/data/Q5_5_1_4_5_5_15_0.5_5_90_2400', "val_dataset.tfrecords")
model_path = os.path.join('/root/autodl-tmp/Code/data/Q5_5_1_4_5_5_15_0.5_5_90_2400', "model_answer.tfrecords")

train_dataset = tf.data.TFRecordDataset(train_path)
val_dataset = tf.data.TFRecordDataset(val_path)
model_answer_dataset = tf.data.TFRecordDataset(model_path)

train_dataset = train_dataset.map(map_func=read_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(map_func=read_dataset, num_parallel_calls=tf.data.AUTOTUNE)
model_answer_dataset = model_answer_dataset.map(map_func=read_model, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(buffer_size=1024).prefetch(buffer_size=tf.data.AUTOTUNE).batch(batch_size)
val_dataset = val_dataset.shuffle(buffer_size=1024).prefetch(buffer_size=tf.data.AUTOTUNE).batch(batch_size)
for element in model_answer_dataset.as_numpy_iterator():
    model_answer_emb = element
model_answer_emb = tf.convert_to_tensor(model_answer_emb)

siamese_network = Siamese(max_length, 
                          encoding_method=1, 
                          embedding_size=embedding_size) # between support set and model answer
# inference_network = Inference(embedding_size, 
#                               encoding_method=1, 
#                               num_classes=num_classes, 
#                               support_shots=training_support_shots, 
#                               query_shots=training_query_shots,
#                               batch_size=batch_size,
#                               num_unlabel_samples=num_unlabel_samples,
#                               num_return_sequences=num_return_sequences)
prototypical_network = Prototypical(batch_size=batch_size,
                                    average=True,
                                    num_classes = num_classes,
                                    n_support = training_support_shots,
                                    n_query = training_query_shots,
                                    encoding_size = 64,
                                    num_unlabel_samples=num_unlabel_samples,
                                    num_return_sequences=num_return_sequences)

# inputs1: support set. [num_classes, support_shots, max_length, embedding_size]
inputs1 = tf.keras.Input(shape=(num_classes, training_support_shots, max_length, embedding_size))
# inputs2: query set. [num_classes, query_shots, max_length, embedding_size]
inputs2 = tf.keras.Input(shape=(num_classes, training_query_shots, max_length, embedding_size))
# inputs3 : unlabel samples. [num_unlabel_samples, num_return_sequences, max_length, embedding_length]
inputs3 = tf.keras.Input(shape=(num_unlabel_samples, (num_return_sequences+1), max_length, embedding_size))
# inputs3: support set label
inputs4 = tf.keras.Input(shape=(num_classes*training_support_shots))

dis_support, dis_query, dis_unlabel = siamese_network(inputs1, inputs2, inputs3, model_answer_emb)

# new_support_dis = inference_network(dis_support, inputs4)
# new_query_dis = inference_network(dis_support, inputs4, unlabel_distance=None, query_distance=dis_query)
# new_unlabel_dis = inference_network(dis_support, inputs4, unlabel_distance=dis_unlabel, query_distance=None)

# the outputs3 is just
outputs1, outputs2, outputs3 = prototypical_network(dis_support, dis_query, dis_unlabel)
# outputs1, outputs2, outputs3 = prototypical_network(new_support_dis, new_query_dis, new_unlabel_dis)
model = tf.keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=[outputs1, outputs2, outputs3])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)
# Instantiate an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_loss_tracker = tf.keras.metrics.Mean('loss')
train_accuracy_tracker = tf.keras.metrics.Mean('acc')

val_loss_tracker = tf.keras.metrics.Mean('loss')
val_accuracy_tracker = tf.keras.metrics.Mean('acc')

@tf.function
def exp_inner_product(x, y, temperature_factor):
    inner_product = tf.reduce_sum(tf.multiply(x, y))
    exp = tf.math.exp(tf.math.divide(inner_product, temperature_factor))
    return exp

@tf.function
def contrast_loss(unlabel):
    encoding_size = unlabel.shape[-1]
    random_answer = tf.slice(unlabel, [0,0,0,0], [batch_size, num_unlabel_samples, 1, encoding_size])
    # [batfch_size, num_unlabel_samples, num_return_sequencecs, encoding_size] 
    paraphrase_answer = tf.slice(unlabel, [0,0,1,0],[batch_size, num_unlabel_samples, num_return_sequences, encoding_size])

    num_paraphrase_sample = 1
    new_paraphrase_sample = tf.slice(paraphrase_answer, [0,0,4,0], [batch_size, num_unlabel_samples, num_paraphrase_sample, encoding_size])

    # unlabel data with data augmentation for contrast learning
    contrast_unlabel = tf.concat([random_answer, new_paraphrase_sample], axis=2)

    temperature_factor = 20.0
    loss = tf.constant(0.)
    for batch in tf.range(batch_size):
        temp_contrast_unlabel = contrast_unlabel[batch]
        loss_per_batch = tf.constant(0.)
        for i in tf.range(num_unlabel_samples):
            # num of matched instances per unlabel sample
            matched_instance = temp_contrast_unlabel[i]
            matched_metric = exp_inner_product(matched_instance[0], matched_instance[1], temperature_factor)
            for j in tf.range(num_paraphrase_sample+1):
                unmatched_metric = tf.constant(0.)
                left_unmatched_instances = tf.gather(temp_contrast_unlabel, tf.range(i))
                right_unmatched_instancecs = tf.gather(temp_contrast_unlabel, tf.range(i+1,num_unlabel_samples))
                unmatched_instance = tf.concat([left_unmatched_instances, right_unmatched_instancecs], axis=0)
                unmatched_instance = tf.reshape(unmatched_instance, [-1, encoding_size])
                for k in tf.range((num_unlabel_samples-1)*(num_paraphrase_sample+1)):
                    unmatched_metric = tf.math.add(unmatched_metric, exp_inner_product(unmatched_instance[k], matched_instance[j], temperature_factor))
                loss_instance = tf.math.log(tf.math.divide(matched_metric, tf.math.add(matched_metric, unmatched_metric)))
                loss_per_batch = tf.math.add(loss_per_batch, loss_instance)
        loss_per_batch = tf.math.divide(loss_per_batch, num_unlabel_samples*(num_paraphrase_sample+1))
        loss = tf.math.add(loss, loss_per_batch)
    
    return -loss

@tf.function
def train_step(x, y, epoch):
    with tf.GradientTape() as tape:
        # log_p_y: [batch_size, num_classes, query_shots, num_classes]
        # log_p  : [batch_size, num_unlabel_samples, num_unlabel_samples]
        # unlabel: [batch_size, num_unlabel_samples, num_return_sequences+1, encoding_size]
        dists_label, dists_unlabel, unlabel = model(x, training=True)

        log_p_y = tf.nn.log_softmax(-dists_label, axis=-1)
        log_p_y = tf.reshape(log_p_y, [batch_size, num_classes, training_query_shots, -1])
        log_p = tf.nn.softmax(-dists_unlabel, axis=-1)

        # return the max index, not actual label
        y_pred = tf.reshape(tf.cast(tf.argmax(log_p_y, axis=-1), tf.float32), shape=[batch_size, -1])   # we want log_p_y as big as possible
        y_hat = tf.cast(tf.reshape(y, [batch_size, num_classes, training_query_shots]), tf.int32)
        # shape: [batch_size, num_classes, query_shots, num_classes]
        y_onehot = tf.cast(tf.one_hot(y_hat, num_classes), tf.float32)

        # supervised cross entropy loss
        loss1 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        beta = 0.1
        # unsupervised entropy loss
        loss2 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(log_p, tf.math.log(log_p+beta)), axis=-1), [-1]))
        # unsupervised contrast loss
        loss3 = contrast_loss(unlabel)

        # alpha is a hyperparameter
        alpha = 0.5
        gamma = 0.002
        t = (epochs-epoch) / epochs  # 递减函数
        weight = tf.math.pow(t, alpha)
        loss = tf.math.multiply(weight,loss1) + tf.math.multiply((1-weight), loss2) + tf.math.multiply(gamma, loss3)
        
        eq = tf.cast(tf.equal(y_pred, 
                              tf.cast(y, tf.float32)), tf.float32)
        acc = tf.reduce_mean(eq)
    # compute gradients
    trainable_vars = model.trainable_variables   
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    
    train_loss_tracker.update_state(loss)
    train_accuracy_tracker.update_state(acc)
    return train_loss_tracker.result(), train_accuracy_tracker.result(), loss1, loss2, loss3

@tf.function
def test_step(x, y, epoch):
    # Compute predictions
    dists_label, dists_unlabel, unlabel = model(x, training=False)
    log_p_y = tf.nn.log_softmax(-dists_label, axis=-1)
    log_p_y = tf.reshape(log_p_y, [batch_size, num_classes, training_query_shots, -1])
    log_p = tf.nn.softmax(-dists_unlabel, axis=-1)

    y_pred = tf.reshape(tf.cast(tf.argmax(log_p_y, axis=-1), tf.float32), shape=[batch_size, -1]) # we want log_p_y as big as possible
    y_hat = tf.cast(tf.reshape(y, [batch_size, num_classes, training_query_shots]), tf.int32)
    # shape: [batch_size, num_classes, query_shots, num_classes]
    y_onehot = tf.cast(tf.one_hot(y_hat, num_classes), tf.float32)

    loss1 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))    
    beta = 0.1
    loss2 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(log_p, tf.math.log(log_p+beta)), axis=-1), [-1]))
    # loss2 = -tf.reduce_mean(tf.reshape(tf.reduce_mean(tf.multiply(log_p, tf.math.log_sigmoid(log_p)), axis=-1), [-1]))
    loss3 = contrast_loss(unlabel)

    alpha = 0.5
    gamma = 0.001
    t = (epochs-epoch) / epochs
    weight = tf.math.pow(t, alpha)
    # weight = 0.1
    loss = tf.math.multiply(weight,loss1) + tf.math.multiply((1-weight), loss2) + tf.math.multiply(gamma, loss3)
    eq = tf.cast(tf.equal(y_pred, 
                          tf.cast(y, tf.float32)), tf.float32)
    acc = tf.reduce_mean(eq)
    
    val_loss_tracker.update_state(loss)
    val_accuracy_tracker.update_state(acc)

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '/root/tf-logs/' + current_time + '/train'
    test_log_dir = '/root/tf-logs/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    for eposide, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss, accuracy, loss1, loss2, loss3 = train_step(x_batch_train, y_batch_train, epoch)

        if eposide % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (eposide, float(loss))
            )
            print(
                "Training accuracy (for one batch) at step %d: %.4f"
                % (eposide, float(accuracy))
            )
            print("Seen so far: %d samples" % ((eposide + 1) * batch_size))
            print("loss1: {}, loss2: {}, loss3: {}".format(loss1, loss2, loss3))
            
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss_tracker.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy_tracker.result(), step=epoch)
    
    # Display metrics at the end of each epoch.
    train_acc = train_accuracy_tracker.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_accuracy_tracker.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val, epoch)
       
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss_tracker.result(), step=epoch)
        tf.summary.scalar('accuracy', val_accuracy_tracker.result(), step=epoch)

    val_acc = val_accuracy_tracker.result()
    val_accuracy_tracker.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))   
    print('-----------------------------------------------------')
