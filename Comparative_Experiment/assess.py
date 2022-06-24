import pandas as pd
import matplotlib.pyplot as plt
import nltk## for language detection
import numpy as np
from numpy.linalg import norm
from itertools import chain, count
import tensorflow_hub as hub
from sklearn.metrics import cohen_kappa_score
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
    text = str(text).lower().strip()
            
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

def cosine_similarity(A, B):
    # adding a constant to avoid nan
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

if __name__ == "__main__":
    train = pd.read_csv("~/autodl-tmp/Data/asap-sas/train.tsv",sep='\t')
    question_set = 5
    train = train[train.EssaySet==question_set]
    train["EssayText_clean"] = train["EssayText"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=False, lst_stopwords=None))
    model_answer = ["mRNA exits nucleus via nuclear pore",
                        "mRNA travels through the cytoplasm to the ribosome or enters the rough endoplasmic reticulum",
                        "mRNA bases are read in triplets called codons (by rRNA)",
                        "tRNA carrying the complementary (U=A, C+G) anticodon recognizes the complementary codon of the mRNA",
                        "The corresponding amino acids on the other end of the tRNA are bonded to adjacent tRNA’s amino acids",
                        "A new corresponding amino acid is added to the tRNA",
                        "Amino acids are linked together to make a protein beginning with a START codon in the P site (initiation)",
                        "Amino acids continue to be linked until a STOP codon is read on the mRNA in the A site (elongation and termination)"]

    # load Universal Sentence Encoder model
    embed = hub.load("https://hub.tensorflow.google.cn/google/universal-sentence-encoder/4")

    # sentence level tokenization 
    answer_vec = []
    for answer in train['EssayText_clean'].values:
        sentences = nltk.tokenize.sent_tokenize(answer)
        sentences_vec = embed(sentences).numpy().tolist()
        answer_vec.append(sentences_vec)

    model_answer_vec = embed(model_answer).numpy().tolist()

    # compute similarity matrix
    similarity_tensor = []
    for answer in answer_vec:
        similarity_matrix = []
        for model_answer_sentence in model_answer_vec:
            row = []
            for answer_sentence in answer:
                similarity = cosine_similarity(model_answer_sentence, answer_sentence)
                row.append(similarity)
            similarity_matrix.append(row)
        similarity_tensor.append(similarity_matrix)


    threshold=0.4
    # entire answer comparison
    entire_model_answer = '. '.join(model_answer)
    emb_entire_model_answer = embed([entire_model_answer])
    similarity_score = []
    for answer, similarity_matrix in zip(train['EssayText_clean'].values, similarity_tensor):
        useful_sentence_index = np.where(np.array(similarity_matrix).max(axis=0) > 0.4)[0]
        sentences = nltk.tokenize.sent_tokenize(answer)
        # remove useless sentences
        cleaned_answer = ""
        for i in useful_sentence_index:
            cleaned_answer = cleaned_answer + " " + sentences[i]
        emb_cleaned_answer = embed([cleaned_answer])
        similarity_score.append(cosine_similarity(emb_entire_model_answer[0], emb_cleaned_answer[0]))

    a = 1
    m = 4 # according to the rubic range / question set
    grade = a * np.array(similarity_score) * m

    grade = np.rint(np.where(grade<0,0,grade))
    y = np.array(train['Score1'])
    accuracy = (y == grade).sum() / y.shape[0]

    print(f"Accuracy: {accuracy}")
    print(f"QWK: {cohen_kappa_score(grade, y, weights='quadratic')}")