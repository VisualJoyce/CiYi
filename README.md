![CiYi](cy_logo.png)
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

## PIE Classification

## SemEval 2022 Task 2
Multilingual Idiomaticity Detection and Sentence Embedding

### Data Preprocess
```shell
python experiments/semeval-2022_task02_idiomacity/create_data.py --input_location ../SemEval_2022_Task2-idiomaticity/SubTaskA/Data --output_location data/annotations/semeval-2022_task02_idiomacity/subtask_a
```

### Subtask A

```shell
TRANSFORMER_LAYER=12 ANNOTATION_DIR=data/annotations/semeval-2022_task02_idiomacity/subtask_a/ZeroShot \
MODEL_NAME=bert-base-multilingual-cased LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib \
allennlp predict data/output/semeval-2022_task02_idiomacity/SubTaskA/zero_shot/finetune/model.tar.gz \
data/annotations/semeval-2022_task02_idiomacity/subtask_a/ZeroShot/eval.jsonl \
--predictor semeval-2022_task02_idiomacity_subtask_a \
--output-file data/output/semeval-2022_task02_idiomacity/SubTaskA/zero_shot/finetune/eval_predict.csv \
--include-package ciyi --cuda-device 0
```

### Subtask B

```shell
python experiments/semeval-2022_task02_idiomacity/subtask_b/create_data.py \
--output_location data/annotations/semeval-2022_task02_idiomacity/subtask_b \
--sts_dataset_path stsbenchmark.tsv.gz  --input_location ../SemEval_2022_Task2-idiomaticity/SubTaskB
```

```shell
TRANSFORMER_LAYER=12 ANNOTATION_DIR=data/annotations/semeval-2022_task02_idiomacity/subtask_b/pretrain \
MODEL_NAME=bert-base-multilingual-cased LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib \
allennlp train experiments/semeval-2022_task02_idiomacity/subtask_b/pretrain.jsonnet \
-s data/output/semeval-2022_task02_idiomacity/SubTaskB/pretrain/bert-base-multilingual-cased \
--include-package ciyi
```

## Acknowledgement
We recommend the following repos:
* [lexcomp](https://github.com/vered1986/lexcomp)