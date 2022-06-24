import sys
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

class Dataset:
    '''
    This class will facilitate the creation of a few-shot dataset
    '''
    def __init__(self, question_set, set=None, random_state=233):
        data = pd.read_csv("~/autodl-tmp/Data/train.tsv",sep='\t')
        ds = data[data.EssaySet==question_set]
        answer = list(ds.EssayText)
        score = list(ds.Score1)

        # if set=='train':
        #     answer,_,score,_ = train_test_split(answer, score, test_size=0.3, random_state=random_state)
        #     answer,_,score,_ = train_test_split(answer, score, test_size=0.3, random_state=random_state)
        # elif set=='test':
        #     _,answer,_,score = train_test_split(answer, score, test_size=0.3, random_state=random_state)
        # elif set=='validation':
        #     answer,_,score,_ = train_test_split(answer, score, test_size=0.3, random_state=random_state)
        #     _,answer,_,score = train_test_split(answer, score, test_size=0.3, random_state=random_state)

        if set=='train':
            answer,_,score,_ = train_test_split(answer, score, test_size=0.2, random_state=random_state)
        elif set=='validation':
            _,answer,_,score = train_test_split(answer, score, test_size=0.2, random_state=random_state)
        
        self.data = {}
        for student_answer, label in zip(answer, score):
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(student_answer)
            
        self.labels = list(self.data.keys())
    
    # get on task dataset: n-way, k shot
    def get_mini_dataset(self, shots, num_classes, split=False, query_shots=None):
        temp_labels = np.zeros(shape=(num_classes * shots))                    # needed labels for one task
        temp_answer = np.empty(shape=(num_classes * shots, 1), dtype='object') # n-way, k-shot
        if split:
            test_labels = np.zeros(shape=(num_classes * query_shots))                      # query set label
            test_answer = np.empty(shape=(num_classes * query_shots, 1), dtype='object')  # n-way, one shot: query set
            
        # Get a random subset of labels from the entire label set.
        # number of classes per episode
        label_subset = sorted(random.sample(self.labels, k=num_classes)) 

        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in few shot learning.
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_obj  # support set label

            # If creating a split dataset (query set) for testing, select QUERY_SHOTS extra samples from each
            # label to create the test dataset (query set).
            if split:
                test_labels[class_idx * query_shots : (class_idx + 1) * query_shots] = class_obj
                answer_to_split = random.sample(
                    self.data[label_subset[class_idx]], k=shots + query_shots # 1 is an extra sample
                )

                # assign the last QUERY_SHOTS samples to query set
                test_answer[class_idx * query_shots : (class_idx + 1) * query_shots] = np.array(answer_to_split[-query_shots:]).reshape(query_shots, 1)
                temp_answer[
                    class_idx * shots : (class_idx + 1) * shots
                ] = np.array(answer_to_split[:-query_shots]).reshape(shots, 1) # assign the remaining samples to the support set
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of answers.
                temp_answer[
                    class_idx * shots : (class_idx + 1) * shots
                ] = np.array(random.sample(self.data[label_subset[class_idx]], k=shots)).reshape(shots, 1)  # only support set
        
        if split:
            return temp_answer, test_answer, test_labels
        else:
            return temp_answer
    
    def get_random_sample(self, num_samples):
        # return one dimension list
        return random.sample(list(np.concatenate(list(self.data.values())).flat), num_samples)