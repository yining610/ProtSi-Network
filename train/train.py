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
from decoder import read_dataset, read_model
import datetime
from sklearn.metrics import cohen_kappa_score

print(f'Current working directory: {os.getcwd()}')

# set parameters
from config import *

train_path = os.path.join('/root/autodl-nas/Q10_3_1_3_5_5_15_0.5_5_90_8100', "train_dataset.tfrecords")
val_path = os.path.join('/root/autodl-nas/Q10_3_1_3_5_5_15_0.5_5_90_8100', "val_dataset.tfrecords")
model_path = os.path.join('/root/autodl-nas/Q10_3_1_3_5_5_15_0.5_5_90_8100', "model_answer.tfrecords")

train_dataset = tf.data.TFRecordDataset(train_path)
val_dataset = tf.data.TFRecordDataset(val_path)
model_answer_dataset = tf.data.TFRecordDataset(model_path)

train_dataset = train_dataset.map(map_func=read_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(map_func=read_dataset, num_parallel_calls=tf.data.AUTOTUNE)
model_answer_dataset = model_answer_dataset.map(map_func=read_model, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(buffer_size=1024).prefetch(buffer_size=tf.data.AUTOTUNE).batch(batch_size)
val_dataset = val_dataset.shuffle(buffer_size=1024).prefetch(buffer_size=tf.data.AUTOTUNE).batch(batch_size)
model_answer_dataset = model_answer_dataset.shuffle(buffer_size=1024).prefetch(buffer_size=tf.data.AUTOTUNE).repeat().batch(batch_size)

siamese_network = Siamese(max_length, 
                          embedding_size=embedding_size)

prototypical_network = Prototypical(batch_size=batch_size,
                                    average=True,
                                    num_classes = num_classes,
                                    n_support = training_support_shots,
                                    n_query = training_query_shots,
                                    encoding_size = encoding_size,
                                    num_unlabel_samples=num_unlabel_samples,
                                    num_return_sequences=num_return_sequences)

# inputs1: support set. [num_classes, support_shots, max_length, embedding_size]
inputs1 = tf.keras.Input(shape=(num_classes, training_support_shots, max_length, embedding_size))
# inputs2: query set. [num_classes, query_shots, max_length, embedding_size]
inputs2 = tf.keras.Input(shape=(num_classes, training_query_shots, max_length, embedding_size))
# inputs3 : unlabel samples. [num_unlabel_samples, num_return_sequences+1, max_length, embedding_length]
inputs3 = tf.keras.Input(shape=(num_unlabel_samples, (num_return_sequences+1), max_length, embedding_size))
# inputs4: model answer. [max_length, embedding_size]
inputs4 = tf.keras.Input(shape=(max_length, embedding_size))

dis_support, dis_query, dis_unlabel = siamese_network(inputs1, inputs2, inputs3, inputs4)

outputs1, outputs2, outputs3 = prototypical_network(dis_support, dis_query, dis_unlabel)
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
train_qwk_tracker = tf.keras.metrics.Mean('QWK')

val_loss_tracker = tf.keras.metrics.Mean('loss')
val_accuracy_tracker = tf.keras.metrics.Mean('acc')
val_qwk_tracker = tf.keras.metrics.Mean('qwk')


@tf.function
def exp_inner_product(x, y, temperature_factor):
    inner_product = tf.reduce_sum(tf.multiply(x, y))
    exp = tf.math.exp(tf.math.divide(inner_product, temperature_factor))
    return exp

@tf.function
def contrast_loss(unlabel):
    '''
    Compute contrastive loss for the similarity matrix computed from siamese network

    Inputs
        unlabel: [batch_size*num_unlabel_samples*(num_return_sequences+1), 32]

    Output:
        loss: float
    '''
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
        # p  : [batch_size, num_unlabel_samples, num_unlabel_samples]
        # unlabel: [batch_size, num_unlabel_samples, num_return_sequences+1, encoding_size]
        dists_label, dists_unlabel, unlabel = model(x, training=True)

        log_p_y = tf.nn.log_softmax(-dists_label, axis=-1)
        log_p_y = tf.reshape(log_p_y, [batch_size, num_classes, training_query_shots, -1])
        p = tf.nn.softmax(-dists_unlabel, axis=-1)

        # return the max index, not actual label
        y_pred = tf.reshape(tf.cast(tf.argmax(log_p_y, axis=-1), tf.float32), shape=[batch_size, -1])   # we want log_p_y as big as possible
        y_hat = tf.cast(tf.reshape(y, [batch_size, num_classes, training_query_shots]), tf.int32)
        # shape: [batch_size, num_classes, query_shots, num_classes]
        y_onehot = tf.cast(tf.one_hot(y_hat, num_classes), tf.float32)

        # supervised cross entropy loss
        loss1 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        # unsupervised entropy loss
        loss2 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(p, tf.math.log(p+beta)), axis=-1), [-1]))
        # unsupervised contrast loss
        loss3 = contrast_loss(unlabel)

        
        t = (epochs - epoch) / (epochs + n)
        weight = tf.math.pow(t, alpha)
        weighted_loss1 = tf.math.multiply(weight,loss1)
        weighted_loss2 = tf.math.multiply((1-weight), loss2)
        weighted_loss3 = tf.math.multiply(gamma, loss3) 
        loss =  weighted_loss1 + weighted_loss2 + weighted_loss3
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
    return train_loss_tracker.result(), train_accuracy_tracker.result(), y_pred, y_hat, weighted_loss1, weighted_loss2, weighted_loss3

@tf.function
def test_step(x, y, epoch):
    # Compute predictions
    dists_label, dists_unlabel, unlabel = model(x, training=False)
    log_p_y = tf.nn.log_softmax(-dists_label, axis=-1)
    log_p_y = tf.reshape(log_p_y, [batch_size, num_classes, training_query_shots, -1])
    p = tf.nn.softmax(-dists_unlabel, axis=-1)

    y_pred = tf.reshape(tf.cast(tf.argmax(log_p_y, axis=-1), tf.float32), shape=[batch_size, -1]) # we want log_p_y as big as possible
    y_hat = tf.cast(tf.reshape(y, [batch_size, num_classes, training_query_shots]), tf.int32)
    # shape: [batch_size, num_classes, query_shots, num_classes]
    y_onehot = tf.cast(tf.one_hot(y_hat, num_classes), tf.float32)

    loss1 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))    
    loss2 = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(p, tf.math.log(p+beta)), axis=-1), [-1]))
    loss3 = contrast_loss(unlabel)

    t = (epochs - epoch) / (epochs + n)
    weight = tf.math.pow(t, alpha)
    weighted_loss1 = tf.math.multiply(weight,loss1)
    weighted_loss2 = tf.math.multiply((1-weight), loss2)
    weighted_loss3 = tf.math.multiply(gamma, loss3) 
    loss =  weighted_loss1 + weighted_loss2 + weighted_loss3
    
    eq = tf.cast(tf.equal(y_pred, 
                          tf.cast(y, tf.float32)), tf.float32)
    acc = tf.reduce_mean(eq)
    
    val_loss_tracker.update_state(loss)
    val_accuracy_tracker.update_state(acc)
    return y_pred, y_hat

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for eposide, ((x_batch_train, y_batch_train),  model_answer_batch) in enumerate(zip(train_dataset, model_answer_dataset)):
        x = x_batch_train + (model_answer_batch,)
        loss, accuracy, y_pred, y_hat, loss1, loss2, loss3 = train_step(x, y_batch_train, epoch)
        qwk = cohen_kappa_score(y_pred.numpy().flatten(order='C'), y_hat.numpy().flatten(order='C'), weights='quadratic')
        train_qwk_tracker.update_state(qwk)
        
        if np.isnan([loss1, loss2, loss3]).sum() > 0:
            print(loss1, loss2, loss3)
        
        if eposide % 200 == 0:
            print("loss1: {}, loss2: {}, loss3: {}".format(loss1, loss2, loss3))
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (eposide, float(loss))
            )
            print(
                "Training accuracy (for one batch) at step %d: %.4f"
                % (eposide, float(accuracy))
            )
            print(
                "Training QWK (for one batch) at step %d: %.4f"
                % (eposide, float(qwk))
            )
            print("Seen so far: %d samples" % ((eposide + 1) * batch_size))
            
    # Display metrics at the end of each epoch.
    train_acc = train_accuracy_tracker.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    train_qwk = train_qwk_tracker.result()
    print("Training QWK over epoch: %.4f" % (float(train_qwk),))

    # Reset training metrics at the end of each epoch
    train_accuracy_tracker.reset_states()
    train_qwk_tracker.reset_states()

    prediction_result = []
    y_label = []
    # Run a validation loop at the end of each epoch.
    for (x_batch_val, y_batch_val), model_answer_batch in zip(val_dataset, model_answer_dataset):
        val_x = x_batch_val + (model_answer_batch,)
        y_pred, y_hat = test_step(val_x, y_batch_val, epoch)
        qwk = cohen_kappa_score(y_pred.numpy().flatten(order='C'), y_hat.numpy().flatten(order='C'), weights='quadratic')
        val_qwk_tracker.update_state(qwk)
        
        # the last epoch
        if epoch == (epochs-1):
            prediction_result.append(y_pred)
            y_label.append(y_hat)

    val_acc = val_accuracy_tracker.result()
    val_qwk = val_qwk_tracker.result()
    val_accuracy_tracker.reset_states()
    val_qwk_tracker.reset_states()
    
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Validation QWK: %.4f" % (float(val_qwk),))
    print("Time taken: %.2fs" % (time.time() - start_time))   
    print('-----------------------------------------------------')

# model.save('saved_model/model_10')
np.save('Q10_prediction.npy', prediction_result)
np.save('Q10_label.npy', y_label)