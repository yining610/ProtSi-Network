import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk## for language detection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import wordnet
from random import randint
from sklearn.metrics import cohen_kappa_score

import numpy as np
from numpy.linalg import norm

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    '''
    Preprocess a string.
    :parameter
        :param text: string - name of column containing text
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    '''
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def WordNet(corpus):
    # apply wordnet to corpus
    new_corpus = pd.Series(dtype='string')
    for answer in corpus:
        # Get the list of words from the entire text
        words = nltk.tokenize.word_tokenize(answer)

        # Identify the parts of speech
        tagged = nltk.pos_tag(words)

        output = ""

        for i in range(0,len(words)):
            replacements = []

            # Only replace nouns with nouns, vowels with vowels etc.
            for syn in wordnet.synsets(words[i]):

                # Do not attempt to replace proper nouns or determiners
                if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
                    break
                
                # The tokenizer returns strings like NNP, VBP etc
                # but the wordnet synonyms has tags like .n.
                # So we extract the first character from NNP ie n
                # then we check if the dictionary word has a .n. or not 
                word_type = tagged[i][1][0].lower()
                if syn.name().find("."+word_type+"."):
                    # extract the word only
                    r = syn.name()[0:syn.name().find(".")]
                    replacements.append(r)

            if len(replacements) > 0:
                # Choose a random replacement
                replacement = replacements[randint(0,len(replacements)-1)]
                output = output + " " + replacement
            else:
                # If no replacement could be found, then just use the
                # original word
                output = output + " " + words[i]
        output = pd.Series(output)
        new_corpus = pd.concat([new_corpus,output], axis=0)
        # new_corpus = new_corpus.append(output)
    return new_corpus

def cosine_similarity(A, B):
    # adding a constant to avoid nan
    cosine = np.dot(A,B)/(norm(A+0.001)*norm(B+0.001))
    return cosine

def compute_cosine_matrix(input, train_dataset):
    cosine_matrix = []
    for i in range(len(input)):
        cosine_vec = []
        for j in range(len(train_dataset)):
            if i != j:
                cosine_vec.append(cosine_similarity(input.values[:,0:-1][i], train_dataset.values[:,0:-1][j]))
            else:
                cosine_vec.append(0)
        cosine_matrix.append(cosine_vec)
    return cosine_matrix

if __name__ == "__main__":
    train = pd.read_csv("~/autodl-tmp/Data/asap-sas/train.tsv",sep='\t')

    question_set = 10

    train = train[train.EssaySet==question_set]
    # preprocess
    lst_stopwords = nltk.corpus.stopwords.words("english")
    train["EssayText_clean"] = train["EssayText"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))
    corpus = train["EssayText_clean"]

    # Apply wordnet
    new_corpus = WordNet(corpus)
    new_corpus = new_corpus.reset_index(drop=True)
    lst_tokens = nltk.tokenize.word_tokenize(new_corpus.str.cat(sep=" "))
    vocabulary = set(lst_tokens)

    # term frequency matrix
    vectorizer = CountVectorizer(vocabulary=vocabulary, min_df=0,
                             stop_words=frozenset(), token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(train['EssayText_clean'].values)
    FC_train = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())

    y = train['Score1'].values

    # calculate information gain
    info_gain = dict(zip(vectorizer.get_feature_names(),
               mutual_info_classif(FC_train, y, discrete_features=True)
               ))

    K = 15
    top_items = dict(sorted(info_gain.items(), key=lambda x:x[1], reverse=True)[:K])
    selected_terms = list(top_items.keys())

    FC_K_train = FC_train.loc[:,selected_terms]
    FC_K_train['Score1'] = y

    train_dataset, val_dataset  = train_test_split(FC_K_train, test_size=0.2, random_state=233)

    # compute cosine similarity
    train_cosine_matrix = compute_cosine_matrix(train_dataset, train_dataset)
    val_cosine_matrix = compute_cosine_matrix(val_dataset, train_dataset)

    # Apply KNN
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(train_cosine_matrix, train_dataset['Score1'])
    print(f"Accuracy: {knn.score(val_cosine_matrix, val_dataset['Score1'])}")

    print(f"QWK: {cohen_kappa_score(knn.predict(val_cosine_matrix), val_dataset['Score1'], weights='quadratic')}")


    