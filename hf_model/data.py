# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
 
# import pandas as pd 
# from transformers import AutoModel, AutoTokenizer
# from torch.utils.data import Dataset, DataLoader 

# model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
# tokenizer = AutoTokenizer.from_pretrained(model)


def load_pairs(dataset_name, split, load_percent, num_shards, 
               features, special_chars, load_from, scramble, save_tsv):

    tsv_filename = "./" + dataset_name + "_" +split + "_" + \
                   str(load_percent) + "_" + str(num_shards) + ".tsv"

    if load_from == "tfds":
        import tensorflow_datasets as tfds 
        print ("Loading data. If the data not available locally, download first.")

        dataset = tfds.load(name=dataset_name, download=True, split=
                split+ '[{}%:{}%]'.format(0, load_percent)
                )

        if scramble: 
            dataset.shuffle(4096)

        dataset = dataset.shard(num_shards=num_shards, index=0)

    #    plain_pairs = [(piece[features[0]], piece[features[1]]) for piece in dataset]

        pairs = [(normalize_sentence(piece[features[0]].numpy().decode("utf-8"), special_chars), 
                  normalize_sentence(piece[features[1]].numpy().decode("utf-8"), special_chars) )
                  for piece in dataset]

        if save_tsv and load_from == "tfds":
            with open(tsv_filename, 'w') as f:
                for (_doc, _sum) in pairs:
                    f.write("\t".join([_doc, _sum]))
                    f.write("\n")
                                   
    elif load_from == "tsv":
        pairs = []
        with open(tsv_filename, 'r') as f:
            for line in f:
                try:
                    [_doc, _sum] = line.replace("\n","").split("\t")
                    pairs.append((_doc, _sum))
                except ValueError:
                    print ("skipping this line:", line)
    return pairs 













# inputs = tokenizer("I loved reading the Hunger Games!")
# inputs


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# tokenized_datasets = tokenizer.map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))








# text = "Here is the sentence I want embeddings for."
# marked_text = "[CLS] " + text + " [SEP]"

# # Tokenize our sentence with the BERT tokenizer.
# tokenized_text = tokenizer.tokenize(marked_text)

# # Print out the tokens.
# print (tokenized_text)

# Defining some key variables 
# MAX_LEN = 512
# TRAIN_BATCH_SIZE = 4
# VALID_BATCH_SIZE = 4
# EPOCHS = 1
# LEARNING_RATE = 1e-05 

# Create a Data Loader Class
# class CNNDailyMailData(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len):
#         self.len = len(dataframe)
#         self.data = dataframe
#         self.tokenizer = tokenizer
#         self.max_len = max_len
        
#     def __getitem__(self, index):
#         sentence = str(self.data.iloc[index].sents)
#         sentence = " ".join(sentence.split())

#         document = str(self.data.iloc[index].docs)
#         document = " ".join(document.split())

#         inputs = self.tokenizer.batch_encode_plus(
#             [sentence, document], 
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding="max_length",
#             return_token_type_ids=True,
#             truncation=True
#         )
#         ids = inputs['input_ids']
#         mask = inputs['attention_mask']

#         return {
#             'sent_ids': torch.tensor(ids[0], dtype=torch.long),
#             'doc_ids': torch.tensor(ids[1], dtype=torch.long),
#             'sent_mask': torch.tensor(mask[0], dtype=torch.long),
#             'doc_mask': torch.tensor(mask[1], dtype=torch.long),
#             'targets': torch.tensor([self.data.iloc[index].y], dtype=torch.long)
#         } 
    
#     def __len__(self):
#         return self.len



# training_set = CNNDailyMailData(small_train_dataset, tokenizer, MAX_LEN)
# testing_set = CNNDailyMailData(small_eval_dataset, tokenizer, MAX_LEN)

# train_params = {'batch_size': TRAIN_BATCH_SIZE,
#                 'shuffle': True,
#                 'num_workers': 0
#                 }

# test_params = {'batch_size': VALID_BATCH_SIZE,
#                 'shuffle': True,
#                 'num_workers': 0
#                 }

# training_loader = DataLoader(training_set, **train_params)
# testing_loader = DataLoader(testing_set, **test_params)

