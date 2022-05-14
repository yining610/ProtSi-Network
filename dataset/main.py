import warnings
warnings.filterwarnings("ignore")

import gensim.downloader as api
from multiprocessing import Queue
from threading import Thread
import multiprocessing as mp
import logging
import pickle
import sys
import os

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertModel

from dataset import Dataset
from utils import *
from worker import worker


if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    train_dataset = Dataset(question_set=1, set='train')
    valid_dataset = Dataset(question_set=1, set='validation')
    whole_dataset = Dataset(question_set=1)
    
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
    if use_bert=='true':
        print('*******Loading Bert*******')
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        bert = BertModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert.to(device)
        model_tokens =  tokenizer(model_answer, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        model_tokens.to(device)
        model_answer = bert(**model_tokens)[0].cpu().detach().numpy()
    else:
        print('*******Loading Glove*******')
        glove = api.load('glove-wiki-gigaword-300')
        model_answer = lemmatize(remove_stops(tokenization(model_answer)))
        model_emb = []
        for word in model_answer:
            try: 
                model_emb.append(glove[word])
            # embedding for typo with all zero vectors
            except:
                model_emb.append(np.zeros(300,))
        model_emb = np.array(model_emb)
        model_emb = np.pad(model_emb, ((0,max_length-len(model_emb)),(0, 0)),'constant')
        model_answer = torch.from_numpy(model_emb)
        model_answer = torch.reshape(model_answer,shape=(1,max_length,300))

    # 初始化任务队列
    result_task = Queue()
    
    # 创建N个生产者（每个生产者1个线程）
    N = 3
    producers = []
    for i in range(N):
        p = mp.Process(target=worker, args=("worker{}".format(i), episodes//N, result_task, train_dataset, valid_dataset, whole_dataset, use_bert, few_shot_config))
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
    tasks_label = np.squeeze(np.array(tasks_label))
    tasks_query= np.squeeze(np.array(tasks_query))
    query_labels = np.squeeze(np.array(query_labels))
    paraphrase_data = np.squeeze(np.array(paraphrase_data))

    val_tasks = np.squeeze(np.array(val_tasks[::5]))
    val_tasks_label = np.squeeze(np.array(val_tasks_label[::5]))
    val_tasks_query = np.squeeze(np.array(val_tasks_query[::5]))
    val_query_labels = np.squeeze(np.array(val_query_labels[::5]))

    parent_dir = '/root/autodl-tmp/Code/data'
    
    new_director = str(few_shot_config['training_support_shots']) + '_' + \
                   str(few_shot_config['training_query_shots']) + '_' + \
                   str(few_shot_config['training_classes']) + '_' + \
                   str(few_shot_config['num_unlabel_samples']) + '_' + \
                   str(few_shot_config['num_return_sequences']) + '_' + \
                   str(few_shot_config['num_beams']) + '_' + \
                   str(few_shot_config['diversity_penalty']) + '_' + \
                   str(few_shot_config['num_beam_groups']) + '_' + \
                   str(few_shot_config['max_length']) + '_' + \
                   str(use_bert) + '_' + \
                   str(episodes)

    new_dir_path = os.path.join(parent_dir, new_director)
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    # save numpy
    np.save(new_dir_path +'/tasks.npy', tasks)
    np.save(new_dir_path +'/tasks_label.npy', tasks_label)
    np.save(new_dir_path +'/tasks_query.npy', tasks_query)
    np.save(new_dir_path +'/query_labels.npy', query_labels)
    np.save(new_dir_path +'/val_tasks.npy', val_tasks)
    np.save(new_dir_path +'/val_tasks_label.npy', val_tasks_label)
    np.save(new_dir_path +'/val_tasks_query.npy', val_tasks_query)
    np.save(new_dir_path +'/val_query_labels.npy', val_query_labels)
    np.save(new_dir_path +'/model_answer.npy', model_answer)
    np.save(new_dir_path +'/paraphrase_data.npy', paraphrase_data)