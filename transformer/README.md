This directory contains few pre-trained models trained with different datasets.
The datasets were generated using `sentence delete` or `word delete` techniques
mentioned in this [paper](https://aclanthology.org/2022.naacl-main.175/).

## Environmet Setup
- `git clone https://github.com/JobayerAhmmed/SueNes.git`
- `cd SueNes`
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python -m spacy download en_core_web_sm`
- `pip install transformers datasets scikit-learn`
- `pip install pyyaml h5py`
- `mkdir exp exp/data exp/result`

## Bert Tiny CNN Daily Mail TensorFlow
Code for the model is in `bert_tiny_cnndm_tf.py` file.
This model is trained from checkpoint found in 
[prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny). 
Data is generated from CNN Daily Mail dataset using 
[SueNes](https://github.com/forrestbao/SueNes).
Only `sentence delete` technique, defined in 
[SueNes paper](https://aclanthology.org/2022.naacl-main.175/),
is used for data generation.
From the generated data, 10% is used for training and 2.5% is for validation.

### Generate Dataset

- `cd pre`
- `python3 sentence_scramble.py`