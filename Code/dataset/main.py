import warnings
warnings.filterwarnings("ignore")

import numpy as np
from multiprocessing import Queue
from threading import Thread
import multiprocessing as mp
import logging
import pickle
import sys
import os

import torch
import tensorflow as tf

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertModel

from dataset import Dataset
from worker import worker

def write_dataset(model_path, 
                  tasks, 
                  tasks_query, 
                  paraphrase_data, 
                  tasks_label, 
                  query_labels,
                  training_classes,
                  training_support_shots,
                  training_query_shots,
                  max_length,
                  embedding_size,
                  num_unlabel_samples,
                  num_return_sequences,
                  eposides):

    with tf.io.TFRecordWriter(model_path) as file_writer:
        for eposide in range(eposides):
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                # embedding data：tensor
                "tasks": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tasks[eposide].tobytes()])),
                "tasks_query": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tasks_query[eposide].tobytes()])),
                "paraphrase_data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[paraphrase_data[eposide].tobytes()])),

                # label data: vector
                "tasks_label": tf.train.Feature(float_list=tf.train.FloatList(value=tasks_label[eposide])),
                "query_labels": tf.train.Feature(float_list=tf.train.FloatList(value=query_labels[eposide])),

                # parameters
                "training_classes": tf.train.Feature(int64_list=tf.train.Int64List(value=[training_classes])),
                "training_support_shots": tf.train.Feature(int64_list=tf.train.Int64List(value=[training_support_shots])),
                "training_query_shots": tf.train.Feature(int64_list=tf.train.Int64List(value=[training_query_shots])),
                "max_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[max_length])),
                "embedding_size": tf.train.Feature(int64_list=tf.train.Int64List(value=[embedding_size])),
                "num_unlabel_samples": tf.train.Feature(int64_list=tf.train.Int64List(value=[num_unlabel_samples])),
                "num_return_sequences": tf.train.Feature(int64_list=tf.train.Int64List(value=[num_return_sequences]))
            })).SerializeToString()
            file_writer.write(record_bytes)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    train_dataset = Dataset(question_set=5, set='train', random_state=233)
    valid_dataset = Dataset(question_set=5, set='validation', random_state=233)
    whole_dataset = Dataset(question_set=5, random_state=233)
    
    from config import *
     
    print(f"The max length of answer is: {max_length}")

    few_shot_config = {'training_support_shots':training_support_shots,
                       'training_query_shots':training_query_shots,
                       'training_classes':training_classes,
                       'num_unlabel_samples':num_unlabel_samples,
                       'num_return_sequences':num_return_sequences,
                       'num_beams':num_beams,
                       'diversity_penalty':diversity_penalty,
                       'num_beam_groups':num_beam_groups,
                       'max_length': max_length
                       }
    
    model_answer= 'how much vinegar was used in each container. what type of vinegar was used in each containe. what materials to test. what size or surface area of materials should be used. how long each sample was rinsed in distilled water. what drying method to use. what size or type of container to use'
    # bert == True
    print('*******Loading Bert*******')
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    model_tokens =  tokenizer(model_answer, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    model_tokens.to(device)
    model_answer = bert(**model_tokens)[0].cpu().detach().numpy()
    embedding_size = 768

    # 初始化任务队列
    result_task = Queue()
    
    device_num = torch.cuda.device_count()
    # 创建N个生产者（每个生产者1个线程）
    N = device_num * 3
    gpu_id = []
    for i in range(device_num):
        gpu_id.extend([i]*3)
        
    
    producers = []
    for i in range(N):
        p = mp.Process(target=worker, args=("worker{}".format(i), episodes//N, result_task, train_dataset, valid_dataset, whole_dataset, few_shot_config, gpu_id[i]))
        producers.append(p)
        p.start()
                
    # 阻塞线程，等待任务完成
    for i in producers:
        i.join()
    
    results = []
    
    for i in producers:
        tmp_name = result_task.get()
        results.append(pickle.load(open(tmp_name,'rb')))
        # todo merge date
        os.remove(tmp_name)# 删除缓存文件

    # prepare dataset
    tasks = []
    tasks_label =[]
    tasks_query = []
    query_labels = []

    val_tasks = []
    val_tasks_label = []
    val_tasks_query = []
    val_query_labels = []

    paraphrase_data = []

    for result in results:
        for i in result:
            tasks.append(i['tasks'])
            tasks_label.append(i['tasks_label'])
            tasks_query.append(i['tasks_query'])
            query_labels.append(i['query_labels'])
            val_tasks.append(i['val_tasks'])
            val_tasks_label.append(i['val_tasks_label'])
            val_tasks_query.append(i['val_tasks_query'])
            val_query_labels.append(i['val_query_labels'])
            paraphrase_data.append(i['paraphrase_data'])
    
    tasks = np.squeeze(np.array(tasks))
    tasks = np.reshape(tasks, [-1, training_classes, training_support_shots, max_length, embedding_size])
    tasks_label = np.squeeze(np.array(tasks_label))

    tasks_query= np.squeeze(np.array(tasks_query))
    tasks_query = np.reshape(tasks_query, [-1, training_classes, training_query_shots, max_length, embedding_size])
    query_labels = np.squeeze(np.array(query_labels))

    val_tasks = np.squeeze(np.array(val_tasks[::5]))
    val_tasks = np.reshape(val_tasks, [-1, training_classes, training_support_shots, max_length, embedding_size])
    val_tasks_label = np.squeeze(np.array(val_tasks_label[::5]))
    
    val_tasks_query = np.squeeze(np.array(val_tasks_query[::5]))
    val_tasks_query = np.reshape(val_tasks_query,  [-1, training_classes, training_query_shots, max_length, embedding_size])
    val_query_labels = np.squeeze(np.array(val_query_labels[::5]))

    paraphrase_data = np.squeeze(np.array(paraphrase_data))
    paraphrase_data = np.reshape(paraphrase_data, [-1, num_unlabel_samples, (num_return_sequences+1), max_length, embedding_size])
    
    parent_dir = '/root/autodl-tmp/Code/data'
    
    new_director = "Q5_" + str(few_shot_config['training_support_shots']) + '_' + \
                   str(few_shot_config['training_query_shots']) + '_' + \
                   str(few_shot_config['training_classes']) + '_' + \
                   str(few_shot_config['num_unlabel_samples']) + '_' + \
                   str(few_shot_config['num_return_sequences']) + '_' + \
                   str(few_shot_config['num_beams']) + '_' + \
                   str(few_shot_config['diversity_penalty']) + '_' + \
                   str(few_shot_config['num_beam_groups']) + '_' + \
                   str(few_shot_config['max_length']) + '_' + \
                   str(episodes)

    new_dir_path = os.path.join(parent_dir, new_director)
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    
    # # save numpy
    # np.save(new_dir_path +'/tasks.npy', tasks)
    # np.save(new_dir_path +'/tasks_label.npy', tasks_label)
    # np.save(new_dir_path +'/tasks_query.npy', tasks_query)
    # np.save(new_dir_path +'/query_labels.npy', query_labels)
    # np.save(new_dir_path +'/val_tasks.npy', val_tasks)
    # np.save(new_dir_path +'/val_tasks_label.npy', val_tasks_label)
    # np.save(new_dir_path +'/val_tasks_query.npy', val_tasks_query)
    # np.save(new_dir_path +'/val_query_labels.npy', val_query_labels)
    # np.save(new_dir_path +'/model_answer.npy', model_answer)
    # np.save(new_dir_path +'/paraphrase_data.npy', paraphrase_data)

    train_path = os.path.join(new_dir_path, 'train_dataset.tfrecords')
    test_path = os.path.join(new_dir_path, 'val_dataset.tfrecords')
    model_path = os.path.join(new_dir_path,'model_answer.tfrecords')

    write_dataset(model_path = train_path,
                  tasks = tasks,
                  tasks_query = tasks_query,
                  paraphrase_data = paraphrase_data,
                  tasks_label = tasks_label,
                  query_labels = query_labels,
                  training_classes = training_classes,
                  training_support_shots = training_support_shots,
                  training_query_shots = training_query_shots,
                  max_length = max_length,
                  embedding_size = embedding_size,
                  num_unlabel_samples = num_unlabel_samples,
                  num_return_sequences = num_return_sequences,
                  eposides = episodes)

    write_dataset(model_path = test_path,
                  tasks = val_tasks,
                  tasks_query = val_tasks_query,
                  paraphrase_data = paraphrase_data,
                  tasks_label = val_tasks_label,
                  query_labels = val_query_labels,
                  training_classes = training_classes,
                  training_support_shots = training_support_shots,
                  training_query_shots = training_query_shots,
                  max_length = max_length,
                  embedding_size = embedding_size,
                  num_unlabel_samples = num_unlabel_samples,
                  num_return_sequences = num_return_sequences,
                  eposides = int(episodes/5))

    with tf.io.TFRecordWriter(model_path) as file_writer:
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            # write data: tensor
            'model_answer': tf.train.Feature(bytes_list=tf.train.BytesList(value=[model_answer.tobytes()])),
            "max_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[max_length])),
            "embedding_size": tf.train.Feature(int64_list=tf.train.Int64List(value=[embedding_size]))
        })).SerializeToString()
        file_writer.write(record_bytes)
        