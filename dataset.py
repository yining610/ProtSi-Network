import pandas as pd
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split

class Dataset:
     # This class will facilitate the creation of a few-shot dataset
    def __init__(self, question_set, training):
        data = pd.read_csv("~/autodl-tmp/data/asap-sas/train.tsv",sep='\t')
        ds = data[data.EssaySet==question_set]
        answer = list(ds.EssayText)
        score = list(ds.Score1)
        # find the length of longest answer
        self.max_length = len(max(answer, key = len))
        if training:
            answer,_,score,_ = train_test_split(answer, score, test_size=0.3, random_state=42)
        else:
            _,answer,_,score = train_test_split(answer, score, test_size=0.3, random_state=42)
        
        self.data = {}
        for student_answer, label in zip(answer, score):
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(student_answer)
            
        self.labels = list(self.data.keys())
    
    # get on task dataset: n-way, k shot
    def get_mini_dataset(self, shots, num_classes, split=False):
        temp_labels = np.zeros(shape=(num_classes * shots))  # needed labels for one task
        temp_answer = np.empty(shape=(num_classes * shots, 1), dtype='object') # n-way, k-shot
        if split:
            test_labels = np.zeros(shape=(num_classes))   # query set label
            test_answer = np.zeros(shape=(num_classes, 1), dtype='object')  # n-way, one shot: query set
            
        # Get a random subset of labels from the entire label set.
        label_subset = sorted(random.sample(self.labels, k=num_classes)) # number of classes per episode
        # print(label_subset)
        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_obj  # support set label
            # If creating a split dataset (query set) for testing, select an extra sample from each
            # label to create the test dataset (query set).
            if split:
                test_labels[class_idx] = class_obj
                answer_to_split = random.sample(
                    self.data[label_subset[class_idx]], k=shots + 1 # 1 is an extra sample
                )
                test_answer[class_idx] = answer_to_split[-1] # assign the last sample to query set
                temp_answer[
                    class_idx * shots : (class_idx + 1) * shots
                ] = np.array(answer_to_split[:-1]).reshape(shots, 1) # assign the remaining samples to the support set
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of answers.
                temp_answer[
                    class_idx * shots : (class_idx + 1) * shots
                ] = np.array(random.sample(self.data[label_subset[class_idx]], k=shots)).reshape(shots, 1)  # only support set
        
        # dataset = tf.data.Dataset.from_tensor_slices(
        #     (temp_answer.astype('str'), temp_labels.astype(np.int32))
        # )
        # dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        if split:
            return temp_answer, temp_labels, test_answer, test_labels
        else:
            return temp_answer, temp_labels