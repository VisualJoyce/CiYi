import argparse
import csv
import gzip
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path

import jsonlines
import pandas as pda
from datasets import load_dataset


def create_pretrain(sts_dataset_path, output_location, languages):
    samples = defaultdict(list)
    if 'EN' in languages:
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = {
                    "language": "EN",
                    "sentence1": row['sentence1'],
                    "sentence2": row['sentence2'],
                    "label": score
                }

                if row['split'] == 'dev':
                    samples['validation'].append(inp_example)
                elif row['split'] == 'test':
                    samples['test'].append(inp_example)
                else:
                    samples['train'].append(inp_example)

    if 'PT' in languages:
        for split in ['train', 'validation', 'test']:
            dataset = load_dataset('assin2', split=split)
            for elem in dataset:
                ## {'entailment_judgment': 1, 'hypothesis': 'Uma criança está segurando uma pistola de água', 'premise': 'Uma criança risonha está segurando uma pistola de água e sendo espirrada com água', 'relatedness_score': 4.5, 'sentence_pair_id': 1}
                score = float(elem['relatedness_score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = {
                    "language": "PT",
                    "sentence1": elem['hypothesis'],
                    "sentence2": elem['premise'],
                    "label": score
                }
                samples[split].append(inp_example)

    for split in samples:
        with jsonlines.open(os.path.join(output_location, 'pretrain', f'{split}.jsonl'), "w") as writer:
            writer.write_all(samples[split])


def _get_train_data(data_location, file_name, include_context, include_idiom):
    file_name = os.path.join(data_location, file_name)

    df = pda.read_csv(file_name, sep=",")
    if include_context:
        df.Previous.fillna('', inplace=True)
        df.Next.fillna('', inplace=True)
        df['sentence'] = df.Previous + df.Target + df.Next
    else:
        df['sentence'] = df.Target

    # ['DataID', 'Language', 'MWE', 'Setting', 'Previous', 'Target', 'Next', 'Label']
    for elem in df.to_dict('records'):
        elem['span'] = elem['MWE']
        elem['label'] = elem['Label']
        elem['lang'] = elem['Language']
        elem['sentence'] = elem['sentence'].replace("\n", " ")
        yield elem


def _get_dev_eval_data(data_location, input_file_name, gold_file_name, include_context, include_idiom):
    # ['ID', 'Language', 'MWE', 'Previous', 'Target', 'Next']
    # ['ID', 'DataID', 'Language', 'Label']
    df = pda.read_csv(os.path.join(data_location, input_file_name),
                      sep=',')
    if not gold_file_name is None:
        df_gold = pda.read_csv(os.path.join(data_location, gold_file_name), sep=",",
                               index_col='ID')
        assert df.shape[0] == df_gold.shape[0]
        df = df.join(df_gold, on='ID', rsuffix='_gold')
    else:
        df['Label'] = 1

    if include_context:
        df.Previous.fillna('', inplace=True)
        df.Next.fillna('', inplace=True)
        df['sentence'] = df.Previous + df.Target + df.Next
    else:
        df['sentence'] = df.Target

    # ['DataID', 'Language', 'MWE', 'Setting', 'Previous', 'Target', 'Next', 'Label']
    for elem in df.to_dict('records'):
        elem['span'] = elem['MWE']
        elem['label'] = elem['Label']
        elem['lang'] = elem['Language']
        elem['sentence'] = elem['sentence'].replace("\n", " ")
        yield elem


def create_finetune(input_location, output_location):
    ## Zero shot data
    with jsonlines.open(os.path.join(output_location, 'ZeroShot', 'train.jsonl'), "w") as writer:
        for item in _get_train_data(
                data_location=input_location,
                file_name='train_zero_shot.csv',
                include_context=True,
                include_idiom=False
        ):
            item['Setting'] = "zero_shot"
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'ZeroShot', 'dev.jsonl'), "w") as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='dev.csv',
                gold_file_name='dev_gold.csv',
                include_context=True,
                include_idiom=False
        ):
            item['Setting'] = "zero_shot"
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'ZeroShot', 'eval.jsonl'), "w") as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='eval.csv',
                gold_file_name=None,  ## Don't have gold evaluation file -- submit to CodaLab
                include_context=True,
                include_idiom=False
        ):
            item['Setting'] = "zero_shot"
            writer.write(item)

    ## OneShot Data (combine both for training)
    with jsonlines.open(os.path.join(output_location, 'OneShot', 'train.jsonl'), 'w') as writer:
        for item in chain(
                _get_train_data(
                    data_location=input_location,
                    file_name='train_zero_shot.csv',
                    include_context=False,
                    include_idiom=True),
                _get_train_data(
                    data_location=input_location,
                    file_name='train_one_shot.csv',
                    include_context=False,
                    include_idiom=True
                )):
            item['Setting'] = "one_shot"
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'OneShot', 'dev.jsonl'), 'w') as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='dev.csv',
                gold_file_name='dev_gold.csv',
                include_context=False,
                include_idiom=True
        ):
            item['Setting'] = "one_shot"
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'OneShot', 'eval.jsonl'), 'w') as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='eval.csv',
                gold_file_name=None,
                include_context=False,
                include_idiom=True
        ):
            item['Setting'] = "one_shot"
            writer.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sts_dataset_path', help='JSON config files')
    # parser.add_argument('--finetune_location', help='JSON config files')
    parser.add_argument('--output_location', help='JSON config files')
    args = parser.parse_args()

    Path(os.path.join(args.output_location, 'pretrain')).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(args.output_location, 'finetune')).mkdir(parents=True, exist_ok=True)

    create_pretrain(args.sts_dataset_path, args.output_location, languages=['EN', "PT"])
    # create_finetune(args.input_location, args.output_location)
