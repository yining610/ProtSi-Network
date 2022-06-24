# meta-training dataset
training_support_shots = 3 # 5
training_query_shots = 1   # 2
training_classes = 3       # based on question set / rubic range

# paraphrase hyperparameters
num_unlabel_samples = 5    # 5
num_return_sequences = 5   # 5
num_beams = 15             # 15
diversity_penalty = 0.5    # 0.5
num_beam_groups = 5        # 5

# model hyperparameter
question_set = 10
episodes = 8100             # train_to_val * device_num * 3 * N
max_length = 90           # max number of tokens per answer: 90 / 50

train_to_val = 3            # train/val