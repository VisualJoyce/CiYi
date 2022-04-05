![image](https://user-images.githubusercontent.com/2136700/161353640-5bb7009d-5d50-4413-a752-f81fdad6a6d0.png)

# CiYi (词义)

A repo for lexical semantics

## MWE Type

## PIE Classification

```bibtex
@inproceedings{tan-jiang-2021-bert,
    title = "Does {BERT} Understand Idioms? A Probing-Based Empirical Study of {BERT} Encodings of Idioms",
    author = "Tan, Minghuan  and
      Jiang, Jing",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)",
    month = sep,
    year = "2021",
    address = "Held Online",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/2021.ranlp-main.156",
    pages = "1397--1407",
    abstract = "Understanding idioms is important in NLP. In this paper, we study to what extent pre-trained BERT model can encode the meaning of a potentially idiomatic expression (PIE) in a certain context. We make use of a few existing datasets and perform two probing tasks: PIE usage classification and idiom paraphrase identification. Our experiment results suggest that BERT indeed can separate the literal and idiomatic usages of a PIE with high accuracy. It is also able to encode the idiomatic meaning of a PIE to some extent.",
}
```

## SemEval 2022 Task 2

Multilingual Idiomaticity Detection and Sentence Embedding

### Subtask A

_Data Preprocess_

```shell
python experiments/semeval-2022_task02_idiomacity/subtask_a/create_data.py \
  --input_location ../SemEval_2022_Task2-idiomaticity/SubTaskA \
  --output_location data/annotations/semeval-2022_task02_idiomacity/subtask_a \
  --phase evaluation
```

_Train_

```shell
bash run_semeval2022_task2a.sh data
```

### Subtask B

_Data Preprocess_

```shell
python experiments/semeval-2022_task02_idiomacity/subtask_b/create_data.py \
  --input_location ../SemEval_2022_Task2-idiomaticity/SubTaskB \
  --output_location data/annotations/semeval-2022_task02_idiomacity/subtask_b \
  --sts_dataset_path stsbenchmark.tsv.gz
```

_Train_

```shell
bash run_semeval2022_task2b.sh data
```

## Acknowledgement

We recommend the following repos:

* [lexcomp](https://github.com/vered1986/lexcomp)
* [allennlp-models](https://github.com/allenai/allennlp-models)
