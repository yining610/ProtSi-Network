# parameters refering to the dataset
# set hyperparameters
num_classes = 3       
max_length = 90       # refer to the file main.py
epochs = 5
embedding_size = 768  # BERT-base is 768
batch_size = 15 
learning_rate = 1e-3

num_unlabel_samples = 5   # refer to the file main.py
num_return_sequences = 5  # refer to the file main.py

encoding_size = 32        # refer to the file siamese.py

# training and testing have the same number of ways and shots and classes
training_support_shots = 3 # refer to the file main.py
training_query_shots = 1  

# training hyperparameters
alpha = 0.8
beta = 0.01
gamma = 0.02
n = 1