import argparse
import csv
import gzip
import os
from collections import defaultdict
from pathlib import Path

import jsonlines
import pandas as pda
from datasets import load_dataset
from tqdm.auto import tqdm


def create_pretrain(sts_dataset_path, output_location, languages):
    samples = defaultdict(list)
    if 'EN' in languages:
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = {
                    "lang": "EN",
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
                    "lang": "PT",
                    "sentence1": elem['hypothesis'],
                    "sentence2": elem['premise'],
                    "label": score
                }
                samples[split].append(inp_example)

    for split in samples:
        with jsonlines.open(os.path.join(output_location, 'pretrain', f'{split}.jsonl'), "w") as writer:
            writer.write_all(samples[split])


def create_predict(input_location, output_location, setting='pretrain'):

    df_dev = pda.read_csv(os.path.join(input_location, 'EvaluationData', 'dev.csv'), sep=",", index_col='ID')
    df_dev_gold = pda.read_csv(os.path.join(input_location, 'EvaluationData', 'dev.gold.csv'), sep=",")
    df = df_dev.join(df_dev_gold, on='ID', rsuffix='_')
    with jsonlines.open(os.path.join(output_location, 'predict', 'dev.jsonl'), "w") as writer:
        for elem in tqdm(df.to_dict('records'), total=df.shape[0]):
            elem['Setting'] = setting
            elem['lang'] = elem['Language']
            writer.write(elem)

    df_eval = pda.read_csv(os.path.join(input_location, 'EvaluationData', 'eval.csv'), sep=",", index_col='ID')
    with jsonlines.open(os.path.join(output_location, 'predict', 'eval.jsonl'), "w") as writer:
        for elem in tqdm(df_eval.to_dict('records'), total=df_eval.shape[0]):
            elem['Setting'] = setting
            elem['lang'] = elem['Language']
            writer.write(elem)

    df_test = pda.read_csv(os.path.join(input_location, 'TestData', 'test.csv'), sep=",", index_col='ID')
    with jsonlines.open(os.path.join(output_location, 'predict', 'test.jsonl'), "w") as writer:
        for elem in tqdm(df_test.to_dict('records'), total=df_test.shape[0]):
            elem['Setting'] = setting
            elem['lang'] = elem['Language']
            writer.write(elem)

    df_finetune = pda.read_csv(os.path.join(input_location, 'TrainData', 'train_data.csv'), sep=",")
    with jsonlines.open(os.path.join(output_location, 'predict', 'finetune.jsonl'), "w") as writer:
        for elem in tqdm(df_finetune.to_dict('records'), total=df_finetune.shape[0]):
            elem['Setting'] = setting
            elem['lang'] = elem['Language']
            writer.write(elem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sts_dataset_path', help='JSON config files')
    parser.add_argument('--input_location', help='JSON config files')
    # parser.add_argument('--finetune_location', help='JSON config files')
    parser.add_argument('--output_location', help='JSON config files')
    args = parser.parse_args()

    Path(os.path.join(args.output_location, 'pretrain')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_location, 'predict')).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(args.output_location, 'finetune')).mkdir(parents=True, exist_ok=True)

    create_pretrain(args.sts_dataset_path, args.output_location, languages=['EN', "PT"])
    create_predict(args.input_location, args.output_location, setting='pretrain')
    # create_finetune(args.input_location, args.output_location)
