import argparse
import csv
import os
from itertools import chain
from pathlib import Path

import jsonlines


def load_csv(path, delimiter=','):
    header = None
    data = list()
    with open(path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            if header is None:
                header = row
                continue
            data.append(row)
    return header, data


def write_csv(data, location):
    with open(location, 'w', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    print("Wrote {}".format(location))
    return


def _get_train_data(data_location, file_name, include_context, include_idiom):
    file_name = os.path.join(data_location, file_name)

    header, data = load_csv(file_name)

    # ['DataID', 'Language', 'MWE', 'Setting', 'Previous', 'Target', 'Next', 'Label']
    for elem in data:
        label = elem[header.index('Label')]
        lang = elem[header.index('Language')]
        sentence1 = elem[header.index('Target')]
        if include_context:
            sentence1 = ' '.join(
                [elem[header.index('Previous')], elem[header.index('Target')], elem[header.index('Next')]])
        sentence2 = elem[header.index('MWE')]
        yield {
            'lang': lang,
            'label': label,
            'sentence': sentence1,
            'span': sentence2
        }


def _get_dev_eval_data(data_location, input_file_name, gold_file_name, include_context, include_idiom):
    input_headers, input_data = load_csv(os.path.join(data_location, input_file_name))
    gold_header = gold_data = None
    if not gold_file_name is None:
        gold_header, gold_data = load_csv(os.path.join(data_location, gold_file_name))
        assert len(input_data) == len(gold_data)

    # ['ID', 'Language', 'MWE', 'Previous', 'Target', 'Next']
    # ['ID', 'DataID', 'Language', 'Label']

    for index in range(len(input_data)):
        label = "1"
        if not gold_file_name is None:
            this_input_id = input_data[index][input_headers.index('ID')]
            this_gold_id = gold_data[index][gold_header.index('ID')]
            assert this_input_id == this_gold_id

            label = gold_data[index][gold_header.index('Label')]

        elem = input_data[index]
        sentence1 = elem[input_headers.index('Target')]
        if include_context:
            sentence1 = ' '.join([elem[input_headers.index('Previous')], elem[input_headers.index('Target')],
                                  elem[input_headers.index('Next')]])

        sentence2 = elem[input_headers.index('MWE')]
        lang = elem[input_headers.index('Language')]
        yield {
            'lang': lang,
            'label': label,
            'sentence': sentence1,
            'span': sentence2
        }


"""
Based on the results presented in `AStitchInLanguageModels' we work with not including the idiom for the zero shot setting and including it in the one shot setting.
"""


def create_data(input_location, output_location):
    ## Zero shot data
    with jsonlines.open(os.path.join(output_location, 'ZeroShot', 'train.jsonl'), "w") as writer:
        for item in _get_train_data(
                data_location=input_location,
                file_name='train_zero_shot.csv',
                include_context=True,
                include_idiom=False
        ):
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'ZeroShot', 'dev.jsonl'), "w") as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='dev.csv',
                gold_file_name='dev_gold.csv',
                include_context=True,
                include_idiom=False
        ):
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'ZeroShot', 'eval.jsonl'), "w") as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='eval.csv',
                gold_file_name=None,  ## Don't have gold evaluation file -- submit to CodaLab
                include_context=True,
                include_idiom=False
        ):
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
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'OneShot', 'dev.jsonl'), 'w') as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='dev.csv',
                gold_file_name='dev_gold.csv',
                include_context=False,
                include_idiom=True
        ):
            writer.write(item)

    with jsonlines.open(os.path.join(output_location, 'OneShot', 'eval.jsonl'), 'w') as writer:
        for item in _get_dev_eval_data(
                data_location=input_location,
                input_file_name='eval.csv',
                gold_file_name=None,
                include_context=False,
                include_idiom=True
        ):
            writer.write(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_location', help='JSON config files')
    parser.add_argument('--output_location', help='JSON config files')
    args = parser.parse_args()

    Path(os.path.join(args.output_location, 'ZeroShot')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_location, 'OneShot')).mkdir(parents=True, exist_ok=True)

    create_data(args.input_location, args.output_location)
