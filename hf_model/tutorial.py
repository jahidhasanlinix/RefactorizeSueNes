from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.optimizers import Adam


dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")

def tokenize_data(data):
    return tokenizer(data["text"], truncation=True)

tokenized_data = dataset.map(tokenize_data, batched=True)
# print(tokenized_data['train'][0])

tf_dataset = model.prepare_tf_dataset(tokenized_data, batch_size=16, shuffle=True, tokenizer=tokenizer)
print(tf_dataset)
print(tf_dataset['train'][0])

model.compile(optimizer=Adam(3e-5))
model.fit(tf_dataset)