# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CNN Dailymail Dataset Modified by SueNes"""


import csv
import json
import os
import datasets


_CITATION = """\
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
"""

_DESCRIPTION = """\
This dataset is created from cnn_dailymail dataset using sentence delete criteria defined in SueNes paper.
"""

_LICENSE = "MIT"

_HOMEPAGE = "https://huggingface.co/datasets/jobayerahmmed/cnn_dailymail_suenes"

_URL = "https://huggingface.co/datasets/jobayerahmmed/cnn_dailymail_suenes/resolve/main/"

# _URLS = {
#     "train": _URL + "train.tsv",
#     "validation": _URL + "validation.tsv",
#     "test": _URL + "test.tsv",
# }

_URLS = {
    "train": "train.tsv",
    "validation": "validation.tsv",
    "test": "test.tsv",
}

class CnnDailymailSuenes(datasets.GeneratorBasedBuilder):
    """CNN Dailymail Dataset Modified by SueNes"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="1.0.0", version=VERSION, description="Initial version"),
    ]

    DEFAULT_CONFIG_NAME = "1.0.0"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "summary": datasets.Value("string"),
                "score": datasets.Value("float32")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # supervised_keys=("sentence", "label"),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # urls = _URLS[self.config.name]
        data_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "filepath": data_files['train'],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_files['validation'],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files['test'],
                },
            )
        ]

    def _generate_examples(self, filepath):
        key = 0
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                splits = [item.strip() for item in line.split('\t')]
                for i in range(2, len(splits), 2):
                    key += 1
                    yield key, {"text": splits[0], "summary": splits[i-1], "score": splits[i]}
                