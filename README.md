# ProtSi: Prototypical-Siamese-Network-for-Few-Shot-Subjective-Answer-Evaluation

# Abstract
Subjective answer evaluation is a time-consuming and tedious task, and the quality of the evaluation is heavily influenced by a variety of subjective personal characteristics. Instead, machine evaluation can effectively assist educators in saving time while also ensuring that evaluations are fair and realistic. However, most existing methods using regular machine learning and natural language processing techniques are generally hampered by a lack of annotated answers and poor model interpretability, making them unsuitable for real-world use. To solve these challenges, we propose ProtSi Network, a unique semi-supervised architecture that for the first time uses few-shot learning to subjective answer evaluation. To evaluate students' answers by similarity prototypes, ProtSi Network simulates the natural process of evaluator scoring answers by combining Siamese Network which consists of BERT and encoder layers with Prototypical Network.  We employed an unsupervised diverse paraphrasing model ProtAugment, in order to prevent overfitting for effective few-shot text classification. By integrating contrastive learning, the discriminative text issue can be mitigated. Experiments on the Kaggle Short Scoring Dataset demonstrate that the ProtSi Network outperforms the most recent baseline models in terms of accuracy and quadratic weighted kappa.

# Contributions
* To the best of our knowledge, our work is the first of its kind to attempt to apply meta-learning to the evaluation of subjective answers.  
* We propose a novel semi-supervised approach, ProtSi Network, for evaluating students' answers from model answers.
* We utilize the data augmentation to deal with overfit and heterogeneous text problems.  
* Through extensive experimental analysis, we demonstrate that the proposed approaches outperform the state-of-the-art on questions from the Kaggle dataset.
# Model Framework
![label](/label.png "The model structure of supervised part of ProtSiNet")
*The framework of supervised part of ProtSiNet*
![unlabel](/unlabel.png "The model structure of unsupervised part of ProtSiNet")
*The framework of unsupervised part of ProtSiNet*
# Requirements
```
pip install -r requirements.txt
```
# Run
## Generate Dataset
```
python /dataset/main.py
```
to prepare few-shot dataset for certain question. The **/dataset/model_answers.txt** file stores all the model answers we used for 10 questions and the question set information and corresponding model answer can be changed in **/dataset/config.py** file.
## Training
```
python /Code/train/train.py
```
to train the model using generated dataset and output testing accuracy and quadratic weighted kappa. The dataset information in **/train/config.py** should be consistent with **/dataset/config.py** 
## Comparative Experiments
The four baseline methods are reimplemented and the code are available in the folder **/Comparatie_Experiment**


