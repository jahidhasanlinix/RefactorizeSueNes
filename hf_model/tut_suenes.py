from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSequenceClassification
import torch

dataset = load_dataset("jobayerahmmed/cnn_dailymail_suenes")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenizer = AutoTokenizer.from_pretrained("cnn_dailymail_suenes")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", num_labels=1)

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The real estate mogul nominated President Obama as well as his sons Eric and Donald Jr to take the challenge next."
sequence_1 = "For a period, Venezuela and Colombia were bitterly at odds. In the past year that relationship has healed. Obstacles to the strength of that relationship remain, analysts say."
sequence_2 = "In the past year that relationship has healed."

# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
# the sequence, as well as compute the attention masks.
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

# paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
# not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]
