from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertModel
import torch
import pickle

def worker(name, episodes, q, train_dataset, valid_dataset, whole_dataset, config, gpu_id):
    '''
    Define worker for multiprocessing
    '''
    print(train_dataset, valid_dataset, whole_dataset, config)
    # 指定子线程cuda
    torch.cuda.set_device(gpu_id)
    
    full_tmp = []
    max_length = config['max_length']
    # fine-trained paraphrase model
    para_model = AutoModelForSeq2SeqLM.from_pretrained("tdopierre/ProtAugment-ParaphraseGenerator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    para_model.to(device)
    para_tokenizer = AutoTokenizer.from_pretrained("tdopierre/ProtAugment-ParaphraseGenerator")

    # bert == true
    print('*******Loading Bert*******')
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    bert.to(device)

    for episode in range(episodes):
        print("dealing with " + str(episode))
        
        new_task, new_task_label, new_query , new_label = train_dataset.get_mini_dataset(shots=config['training_support_shots'], 
                                                                                         num_classes=config['training_classes'], 
                                                                                         split=True, 
                                                                                         query_shots=config['training_query_shots'])
        unlabel_data = whole_dataset.get_random_sample(num_samples=config['num_unlabel_samples'])

        # paraphrase
        batch = para_tokenizer(unlabel_data, return_tensors='pt', padding=True, truncation=True)
        batch.to(device)
        generated_ids  = para_model.generate(batch['input_ids'], num_return_sequences=config['num_return_sequences'], num_beams=config['num_beams'], diversity_penalty=config['diversity_penalty'], num_beam_groups=config['num_beam_groups'])
        generated_sentence = para_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # insert new paraphrase answer into random unlabel answer with fixed step 
        num_return_sequences = config['num_return_sequences']
        paraphrase_result = [None]*(len(unlabel_data)+len(generated_sentence))
        paraphrase_result[::(num_return_sequences+1)] = unlabel_data
        for i in range(num_return_sequences):
            paraphrase_result[(i*(num_return_sequences+1)+1):((i+1)*(num_return_sequences+1)):1] = generated_sentence[(i*num_return_sequences):((i+1)*num_return_sequences):1]
        
        temp = {
            'tasks': [],
            'tasks_label': [],
            'tasks_query': [],
            'query_labels': [],
            'val_tasks': [],
            'val_tasks_label': [],
            'val_tasks_query': [],
            'val_query_labels': [],
            'paraphrase_data': []
        }

        # bert == True
        new_task_tokens = tokenizer(list(new_task.flat), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        new_task_tokens.to(device)
        new_task_emb = bert(**new_task_tokens)[0]
        temp['tasks'].append(new_task_emb.cpu().detach().numpy())
        new_query_tokens = tokenizer(list(new_query.flat), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        new_query_tokens.to(device)
        new_query_emb = bert(**new_query_tokens)[0]
        temp['tasks_query'].append(new_query_emb.cpu().detach().numpy())

        new_paraphrase_tokens = tokenizer(paraphrase_result, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        new_paraphrase_tokens.to(device)
        new_paraphrase_emb = bert(**new_paraphrase_tokens)[0]
        temp['paraphrase_data'].append(new_paraphrase_emb.cpu().detach().numpy())
        temp['tasks_label'].append(new_task_label)
        temp['query_labels'].append(new_label)

        if episode%5==0:
            # validation_tasks / training_tasks = 1 / 5
            #  tasks and tasks_query are used for train
            #  val_tasks and val_tasks_query are used for validation

            new_val_task, new_val_task_label, new_val_query , new_val_label = valid_dataset.get_mini_dataset(shots=config['training_support_shots'], 
                                                                                                             num_classes=config['training_classes'], 
                                                                                                             split=True, 
                                                                                                             query_shots=config['training_query_shots'])

            # bert == true
            new_val_task_tokens = tokenizer(list(new_val_task.flat), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            new_val_task_tokens.to(device)
            new_val_task_emb = bert(**new_val_task_tokens)[0]
            temp['val_tasks'].append(new_val_task_emb.cpu().detach().numpy())
            new_val_query_tokens = tokenizer(list(new_val_query.flat), return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            new_val_query_tokens.to(device)
            new_val_query_emb = bert(**new_val_query_tokens)[0]
            temp['val_tasks_query'].append(new_val_query_emb.cpu().detach().numpy())
            temp['val_tasks_label'].append(new_val_task_label)
            temp['val_query_labels'].append(new_val_label)

        full_tmp.append(temp)
    
    tmp_name="tmp_file_%s"%name
    pickle.dump(full_tmp, open(tmp_name, 'wb'),protocol=3)
    q.put(tmp_name)