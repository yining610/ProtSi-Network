import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
stop_words = stopwords.words('english')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# word level tokenization
def tokenization(answer):
    tokens = []
    txt = re.split(r'(\d+)|[.|,]', str(answer)) # avoid the situation of "content.They" => "contentthey"
    for sub_text in txt:
        sub_text = re.sub(r'[^\w\s]', '', str(sub_text).lower().strip())
        sub_text = sub_text.split()
        tokens = tokens + sub_text
    return tokens

def remove_stops(word_list):
    return [w for w in word_list if w not in stop_words]

def lemmatize(words_list):
    # lemmatize use wordnetlemmatizer
    lemmatized_words = []
    lemmatizer = WordNetLemmatizer()
    for w in words_list:
        lemmatized_words.append(lemmatizer.lemmatize(w))
    # lemmatized_unique_keywords = list(set(lemmatized_keywords))  
    return lemmatized_words

def embedding(task, max_length, model):
    task_emb = []
    for answer in task:
        answer_emb = []
        for word in answer:
            try: 
                answer_emb.append(model[word])
            # embedding for typo
            except KeyError:
                answer_emb.append(np.zeros(300,)) # replace unusual word with zero embedding
        answer_emb = np.array(answer_emb)
        # padding the length
        if answer_emb.size == 0:
            answer_emb = np.zeros(shape=[max_length, 300]) # 300 is the embedding size
        else:
            answer_emb = np.pad(answer_emb, ((0,max_length-len(answer_emb)),(0, 0)),'constant').tolist()
        task_emb.append(answer_emb)
    return np.stack(task_emb, axis=0)

def preprocess(new_task, max_length, model):
    cleaned_task = [tokenization(x) for x  in new_task]
    cleaned_task = [remove_stops(x) for x in cleaned_task]
    cleaned_task = [lemmatize(x) for x in cleaned_task]
    cleaned_task = embedding(cleaned_task, max_length, model)
    return cleaned_task