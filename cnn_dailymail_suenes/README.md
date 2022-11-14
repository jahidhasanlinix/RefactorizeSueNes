---
license: mit
---

# Dataset Card for CNN Dailymail SueNes Dataset

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:**
- **Repository:** [CNN Dailymail SueNes Dataset repository](https://github.com/JobayerAhmmed/cnn_dailymail_suenes_dataset)
- **Paper:** [SueNes: A Weakly Supervised Approach to Evaluating Single-Document Summarization via Negative Sampling](https://aclanthology.org/2022.naacl-main.175.pdf)
- **Leaderboard:**
- **Point of Contact:**

### Dataset Summary

The CNN Dailymail SueNes dataset is generated from CNN Dailymail dataset and 
modified using sentence delete criteria
defined in SueNes paper.

### Supported Tasks and Leaderboards

- 'summarization': This dataset can be used to train a model for abstractive and extractive summarization.

### Languages

English

## Dataset Structure

### Data Instances

For each instance, there is a string for the text, a string for the summary, and a string for the score.

```
{
    "text": "By. Ashley Collman for MailOnline. Donald Trump made the best case that his hair is real by taking the ALS ice-bucket challenge on Wednesday. The real-estate mogul was nominated by Homer Simpson, Vince McMahon and Mike Tyson \u00a0and decided to accept the challenge in style. From the top of Trump Tower, the Donald had the beautiful Miss USA and Miss Universe dump ice-cold bottled water over his head to raise awareness of ALS- also known as Lou Gherig's disease. Scroll down for video. Challenge accepted: Billionaire real-estate mogul Donald Trump took the ALS ice bucket challenge on Thursday, accepting nominations from Homer Simpson, Vince McMahon and Mike Tyson. In the video he even made fun of the ambiguity surrounding his well-coiffed hair. 'Everybody is going crazy over this thing,' he said. 'I guess they want to see whether or not it's my real hair - which it is.' The two beauty-pageant queens then dump the two buckets of water over his head, soaking his bespoke suit. 'What a mess,' Trump says. Doing it in style: Sparing no cost, Trump had only the finest Trump bottled water poured over his head in the challenge. You have 24 hours: The Donald nominated President Obama, as well as his sons Eric and Donald Jr to take the challenge next. Assistants: Miss Universe (left) and Miss USA (right) helped Trump take the challenge. Trump owns the beauty pageant organizations. Lots of glue? The Donald's hair held up well to the challenge, despite a long-standing rumor that he wears a toup\u00e9e. 'What a mess,' Trump said after the two beauty queens dumped the water over his head. The ALS ice-bucket challenge has been sweeping the internet. Those who are challenged can either opt out by donating to the cause, or film themselves dumping ice-water over their heads. Many have chosen to both donate and post a video. While Trump did not post about donating to the charity, the billionaire likely wrote out a check for the good cause. Since July, the ice-bucket challenge has helped raise over $90million to go towards ALS research. Trump went on to nominate President Obama, as well as his sons Eric and Donald Trump Jr. Momentous: The ALS ice-bucket challenge has helped raise over $90million in donations since July.",
    "summary": "The real estate mogul nominated President Obama as well as his sons Eric and Donald Jr to take the challenge next.",
    "score": 1.0
}
```
### Data Fields

- `text`: a string containing the text of a news article
- `summary`: a string containing the summary of the article as written by the article author
- `score`: a string containing the score of the summary

### Data Splits

The CNN Dailymail SueNes dataset has 3 splits: _train_, _validation_, and _test_. Below are the statistics for Version 1.0.0 of the dataset.
| Dataset Split | Number of Instances in Split                |
| ------------- | ------------------------------------------- |
| Train         | 138,502                                     |
| Validation    | 17,917                                      |
| Test          | 17,722                                      |

## Additional Information

### Licensing Information

The CNN Dailymail SueNes dataset version 1.0.0 is released under the [MIT](https://opensource.org/licenses/MIT). 

### Citation Information

```
@inproceedings{bao-etal-2022-suenes,
    title = "{S}ue{N}es: A Weakly Supervised Approach to Evaluating Single-Document Summarization via Negative Sampling",
    author = "Bao, Forrest  and
      Luo, Ge  and
      Li, Hebi  and
      Qiu, Minghui  and
      Yang, Yinfei  and
      He, Youbiao  and
      Chen, Cen",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.175",
    doi = "10.18653/v1/2022.naacl-main.175",
    pages = "2450--2458",
    abstract = "Canonical automatic summary evaluation metrics, such as ROUGE, focus on lexical similarity which cannot well capture semantics nor linguistic quality and require a reference summary which is costly to obtain. Recently, there have been a growing number of efforts to alleviate either or both of the two drawbacks. In this paper, we present a proof-of-concept study to a weakly supervised summary evaluation approach without the presence of reference summaries. Massive data in existing summarization datasets are transformed for training by pairing documents with corrupted reference summaries. In cross-domain tests, our strategy outperforms baselines with promising improvements, and show a great advantage in gauging linguistic qualities over all metrics.",
}
```

### Contributions

Thanks to [@JobayerAhmmed](https://github.com/JobayerAhmmed) and [@jahidhasanlinix](https://github.com/jahidhasanlinix) for adding this dataset.

