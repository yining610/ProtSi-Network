# parameters refering to the dataset
# set hyperparameters
num_classes = 4                    # 如果不是4， prototypical network里有一点bug
max_length =  90      # refer to file dataset.py
epochs = 20
embedding_size = 768  # glove(with preprocecss is 300), BERT(without preprocess is 768)
batch_size = 10
learning_rate = 1e-3
num_unlabel_samples = 5
num_return_sequences = 5

# training and testing have the same number of ways and shots and classes
training_support_shots = 5 # 被修改过了
training_query_shots = 1  # 被修改过了