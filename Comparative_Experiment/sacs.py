import pandas as pd
import numpy as np
from keybert import KeyBERT
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words
import re
import gensim.downloader as api
import math
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.spatial import distance
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import collections
from sklearn.metrics import cohen_kappa_score
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

def correct_typo(word_list):
    # correct typo using Jaccard distance Method
    correct_words = words.words()
    corrected_words = []
    # loop for finding correct spellings
    # based on jaccard distance
    # and printing the correct word
    for word in word_list:
        temp = [(jaccard_distance(set(ngrams(word, 2)),
                                  set(ngrams(w, 2))),w)
                for w in correct_words if w[0]==word[0]]
        if temp != []:
            corrected_words.append(sorted(temp, key = lambda val:val[0])[0][1])
        else:
            continue   
    
    return corrected_words


def lemmatize(words_list):
    # lemmatize use wordnetlemmatizer
    lemmatized_keywords = []
    lemmatizer = WordNetLemmatizer()
    for w in words_list:
        lemmatized_keywords.append(lemmatizer.lemmatize(w))
    lemmatized_unique_keywords = list(set(lemmatized_keywords))  
    return lemmatized_unique_keywords


# remove stop words and tokenize
def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def find_common(multi_list):
    return list(set.intersection(*map(set, multi_list)))

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

if __name__ == "__main__":
    train = pd.read_csv("~/autodl-tmp/Data/asap-sas/train.tsv",sep='\t')

    # For accruacy, keywords can be manually input by evaluator
    ## generate keywords for all 10 answers
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    key_words = {}
    for j in range(1,max(train.EssaySet)+1):
        # Extract keywords using Keybert 
        keywords_list = []
        if train[(train.Score1==3) & (train.EssaySet==j)].empty:
            excellent_answer = train[(train.Score1==2) & (train.EssaySet==j)]
            num_keys = 2
        else:
            excellent_answer = train[(train.Score1==3) & (train.EssaySet==j)]
            num_keys = 3
        for i in range(len(excellent_answer)):
            text = excellent_answer.iloc[i,4]
            temp_keywords = kw_model.extract_keywords(text, 
                                             keyphrase_ngram_range=(1, 1), 
                                             stop_words='english', 
                                             highlight=False,
                                             top_n=num_keys) # how many key words you want per sentence
            temp_keywords_list = list(dict(temp_keywords).keys())
            keywords_list = keywords_list + temp_keywords_list
        unique_keywords = list(set(keywords_list))

        # Correct typo
        corrected_words = correct_typo(unique_keywords)
        # Lemmatization
        lemmatized_unique_keywords = lemmatize(corrected_words)

        key_words[j]=lemmatized_unique_keywords

    solution_sentences_10 = ["Black. The doghouse will be warmer. The black lid made the jar warmest",
                         "Dark gray. The inside will be a little warmer, but not too hot. The dark gray lid increased 6º C more than the white",
                         "Light gray. The inside will stay cooler, but not too cool. The light gray lid was 8º C cooler than the black",
                         "White. The inside will be cooler. The white lid only went up to 42º C"]
    solution_sentences = [solution_sentences_10]

    # Sentence level comparison
    # Split text to sentences using regular expression
    question_list = [10]
    answer_sentences = []
    for i in question_list:
        answer_temp = train[train.EssaySet==i].EssayText
        answer_sentences_temp = []
        for text in answer_temp:
            result = re.split(r'[.]', text.strip())     # tokenize text by period
            result = [item.strip() for item in result]   # strip sentence
            cleaned_result = list(filter(None, result))  # remove empty string
            lowered_result = [x.lower() for x in cleaned_result] # convert to lowercase
            answer_sentences_temp.append(lowered_result)

        answer_sentences.append(answer_sentences_temp)

    # remove stop words and tokenize
    solution_sentences = [list(map(preprocess,solution_sentences_temp)) for solution_sentences_temp in solution_sentences]
    answer_sentences = [[list(map(preprocess, answer_sentence)) for answer_sentence in answer_sentences_temp] for answer_sentences_temp in answer_sentences]
    # remove unusual characters: like "+" and empty string, and single character or number
    solution_sentences = [[list(filter(None, [re.sub('[^a-z0-9]+','', word) for word in solution_sentence])) for solution_sentence in solution_sentences_temp] for solution_sentences_temp in solution_sentences]
    solution_sentences = [[list(filter(None, [word for word in solution_sentence if len(word)> 1])) for solution_sentence in solution_sentences_temp] for solution_sentences_temp in solution_sentences]
    answer_sentences = [[[list(filter(None, [re.sub('.?[^a-z0-9]+.?','', _) for _ in my_list])) for my_list in my_answer] for my_answer in answer_sentences_temp] for answer_sentences_temp in answer_sentences]
    answer_sentences = [[list(filter(None, [list(filter(None, [word for word in my_list if len(word)> 1])) for my_list in my_answer])) for my_answer in answer_sentences_temp] for answer_sentences_temp in answer_sentences]

    model = KeyedVectors.load_word2vec_format(datapath(r"/root/autodl-tmp/Data/GoogleNews-vectors-negative300.bin"), binary=True)

    length = 0
    for question_set in question_list:
        length += len(train[train.EssaySet==question_set])
    print(f"There are total {length} student's answers")

    # using wmd distance to learn semantic similarity
    distance = {}
    for answer_sentences_temp, solution_sentences_temp, question_set in zip(answer_sentences, solution_sentences, question_list):
        data_temp = train[train.EssaySet==question_set]
        for i in range(len(answer_sentences_temp)):
            index = data_temp.iloc[i].Id  # using answer index to build dictionary
            answer = answer_sentences_temp[i]
            all_sentences_dis = []  # nxm matrix: number of answer sentences x number of solution sentences
            for answer_sentence in answer: 
                per_sentence_dis = []   # wmd distance 1 answer sentence -> all solution sentence
                for solution_sentence in solution_sentences_temp:
                    temp_wmd = model.wmdistance(answer_sentence, solution_sentence)
                    if temp_wmd == math.inf: # assign 100 to temp_wmd if it is infinite
                        temp_wmd = 100
                    per_sentence_dis.append(temp_wmd)
                all_sentences_dis.append(per_sentence_dis)
            distance[index] = all_sentences_dis # because each answer has different length, has to use dictionary to store the data

    # Word-level comparison
    # keyword-weight calculation
    # Preprocessing for solution and answer sentences: correct typo and lemmatize
    # keyword_1 = key_words[1]  # for question set 1

    num_keywords_solution = []  # how many keywords in each cleaned solution sentence
    cleaned_solution_sentences = []
    for i,question_set in zip(range(len(question_list)), question_list):
        cleaned_solution_sentences_temp = []
        num_keywords_solution_temp = []
        solution_sentences_temp = solution_sentences[i]
        for solution_sentence in solution_sentences_temp:
            corrected_solution_sentence = correct_typo(solution_sentence)
            lemmatized_solution_sentence = lemmatize(corrected_solution_sentence)
            cleaned_solution_sentences_temp.append(lemmatized_solution_sentence)
            solution_keywords_set = [lemmatized_solution_sentence, key_words[question_set]]
            temp_num_keywords_solution = len(find_common(solution_keywords_set))
            num_keywords_solution_temp.append(temp_num_keywords_solution)        
        num_keywords_solution.append(num_keywords_solution_temp)
        cleaned_solution_sentences.append(cleaned_solution_sentences_temp)

    num_keywords_solution = [[num+0.001 for num in num_keywords_solution_temp] for num_keywords_solution_temp in num_keywords_solution] # avoid division by 0

    keyword_weight = {}

    for answer_sentences_temp, cleaned_solution_sentences_temp, num_keywords_solution_temp, question_set in zip(answer_sentences, cleaned_solution_sentences, num_keywords_solution, question_list):
        data_temp = train[train.EssaySet==question_set]
        for i in range(len(answer_sentences_temp)):
            index = data_temp.iloc[i].Id
            answer = answer_sentences_temp[i]
            all_sentences_weight = []
            for answer_sentence in answer:  # answer_sentence is a word list
                per_sentence_weight = []

                # I think no need to correct typo in student's answer. Number of typo can also influence the final score
                # # Correct typo
                # corrected_answer_sentence = correct_typo(answer_sentence)

                # Lemmatization
                lemmatized_answer_sentence = lemmatize(answer_sentence)
                for j in range(len(cleaned_solution_sentences_temp)):
                    lemmatized_solution_sentence = cleaned_solution_sentences_temp[j]
                    answer_solution_keywords_set = [lemmatized_answer_sentence, lemmatized_solution_sentence,  key_words[question_set]]
                    temp_num_keywords_solution_answer = len(find_common(answer_solution_keywords_set))
                    per_sentence_weight.append(temp_num_keywords_solution_answer/num_keywords_solution_temp[j])
                all_sentences_weight.append(per_sentence_weight)
            keyword_weight[index] = all_sentences_weight

    # without normalization
    # the function is changable
    # use keyword and contextual similarity to predict final score
    ratio = {}
    bias = 4
    for key in distance.keys():
        temp_list = []
        for list1, list2 in zip(distance[key], keyword_weight[key]):
            temp_list.append([i / (j + 0.01) + bias for i, j in zip(list1,list2)])
        ratio[key] = temp_list

    question_index = {}
    for question_set in question_list:
        question_index[question_set] = list(train[train.EssaySet == question_set].Id)

    threshold_ratio = 10 # define the threshold. 
    stat_score = {} # statistical score
    # use 
    for question_set in question_list:
        keys = question_index[question_set]
        stat_score_temp = []
        if (question_set in [1,2,6]):
            max_score = 3 # define the max score
            needed_sentence = 3 # define the requirements

        if question_set == 5:
            max_score = 3
            needed_sentence = 4
        if question_set == 10:
            max_score = 2
            needed_sentence = 3
        if (question_set in [3,4,7,8,9]):
            max_score = 2
            needed_sentence = len(solution_sentences[0])

        grade_per_sentence = max_score / needed_sentence
        for key in keys:
            temp_ratio = ratio[key]
            try:
                true_indices = np.unique(np.transpose((np.array(temp_ratio) < threshold_ratio).nonzero())[:,1])  # retuern the indices of value that below the threshold
            except IndexError:
                true_indices = []
                print(f"Empty answer index: {key}")
            unique_true_indices = np.unique(true_indices)
            predicted_score = round(grade_per_sentence * len(unique_true_indices))  # try ceil or floor function
            if predicted_score > max_score:
                stat_score_temp.append(max_score)
            else:
                stat_score_temp.append(predicted_score)
        stat_score[question_set] = stat_score_temp

    stat_result = []
    for question_set in question_list:
        data_temp = train[train.EssaySet==question_set]
        stat_result_temp = {'Score1':data_temp.Score1,'Score2':data_temp.Score2,'Prediction':stat_score[question_set]}
        stat_result_temp = pd.DataFrame(data=stat_result_temp)
        stat_result_temp['error1'] = abs(stat_result_temp['Score1'] - stat_result_temp['Prediction'])
        stat_result_temp['error2'] = abs(stat_result_temp['Score2'] - stat_result_temp['Prediction'])
        accuracy_temp = np.sum(stat_result_temp['error1'] == 0) / len(stat_result_temp)
        qwk_tmp = cohen_kappa_score(stat_score[question_set], data_temp['Score1'], weights='quadratic')
        print(f"Accuracy for question set {question_set}: {accuracy_temp}")
        print(f"QWK for question set {question_set}: {qwk_tmp}")