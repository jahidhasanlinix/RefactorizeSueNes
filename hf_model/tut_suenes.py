from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSequenceClassification
import torch

dataset = load_dataset("jobayerahmmed/cnn_dailymail_suenes")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenizer = AutoTokenizer.from_pretrained("cnn_dailymail_suenes")
model = AutoModelForSequenceClassification.from_pretrained("cnn_dailymail_suenes", num_labels=1)

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The real estate mogul nominated President Obama as well as his sons Eric and Donald Jr to take the challenge next."
sequence_1 = "For a period, Venezuela and Colombia were bitterly at odds. In the past year that relationship has healed. Obstacles to the strength of that relationship remain, analysts say."
sequence_2 = "In the past year that relationship has healed."

# print special token ids, Added the Special Word [CLS] at the begining and 
# Special word [SEP] at the end
# model_inputs0 = tokenizer(sequence_0)
# print(model_inputs0["input_ids0"])
# tokens= tokenizer.tokenize(sequence_0)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
# print(tokenizer.decode(model_inputs0["input_ids"]))
# print(tokenizer.decode(ids))


# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
# the sequence, as well as compute the attention masks.
token_para = tokenizer(sequence_0, sequence_1, sequence_2, padding=True, truncation=True, return_tensors="pt")
output = model(**token_para)
output_logits = model(**token_para).logits
