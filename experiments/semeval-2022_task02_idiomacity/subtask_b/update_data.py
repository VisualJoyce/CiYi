import argparse
import os

import jsonlines
import pandas as pda


def update_finetune(opts, split):
    predict_file = os.path.join(opts.prediction_location, f"finetune_{split}_predict.csv")
    finetune_original_file = os.path.join(opts.annotation_location, f"{split}_original.jsonl")

    with jsonlines.open(finetune_original_file) as f:
        df_finetune = pda.DataFrame.from_dict([item for item in f])
    columns = df_finetune.columns.tolist()

    df_pretrain = pda.read_csv(predict_file,
                               names=['ID', 'Language', 'Setting', 'Sim'], index_col='ID')

    df_joined = df_finetune.join(df_pretrain, on="ID", rsuffix='_')[columns + ['Sim']]

    data = []
    for item in df_joined.to_dict('records'):
        item['label'] = float(item['Sim'] if item['sim'] == 'None' or pda.isna(item['sim']) else item['sim'])
        data.append(item)

    finetune_file = os.path.join(opts.annotation_location, f"{split}.jsonl")
    with jsonlines.open(finetune_file, "w") as writer:
        writer.write_all(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_location', help='JSON config files')
    parser.add_argument('--prediction_location', help='JSON config files')
    args = parser.parse_args()
    for split in ['train', 'validation']:
        update_finetune(args, split)
