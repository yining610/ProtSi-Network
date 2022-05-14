# meta-training dataset
training_support_shots = 5 # 5
training_query_shots = 2   # 2
training_classes = 3     # 3

# paraphrase hyperparameters
num_unlabel_samples = 5    # 5
num_return_sequences = 5   # 5
num_beams = 15              # 15
diversity_penalty = 0.5    # 0.5
num_beam_groups = 5        # 5

# model hyperparameter
episodes = 2250               # more
max_length = 90             # max number of tokens per answer: 90 / 50
use_bert = 'true'           # Input: True or False. True: Use bert to embedding word. Otherwise use Glove with  word preprocess
                                          # embedding_size: 768 