# ProtSi: Prototypical-Siamese-Network-for-Few-Shot-Subjective-Answer-Evaluation

# Abstract

# Contributions
The following are the main contributions of this paper:
* To the best of our knowledge, our work is the first attempt to apply meta-learning to subject answer evaluation. 
* We propose a novel semi-supervised approach, ProtSi network, for evaluating students' answers from model answers.
* We utilize data augmentation and inference layer to deal with overfit and heterogeneous text problem. 
* We conduct experiments on 4 questions from Kaggle dataset and demonstrate that the proposed methods outperforms the state-of-the-art.

# Model Framework
![label](/label.png "The model structure of supervised part of ProtSiNet")
*The framework of supervised part of ProtSiNet*
![unlabel](/unlabel.png "The model structure of unsupervised part of ProtSiNet")
*The framework of unsupervised part of ProtSiNet*

# Run
```
python /Code/dataset/main.py
```
to generate few-shot dataset\
```
python /Code/train/train.py
```
to train the model and output accuracy


