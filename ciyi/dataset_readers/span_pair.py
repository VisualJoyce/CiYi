import json
import logging
from typing import Dict, Iterable

import numpy as np
import spacy
from Levenshtein import distance
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ListField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from more_itertools import windowed

from ciyi.data.fields.float_field import FloatField

logger = logging.getLogger(__name__)


@DatasetReader.register("span_pair")
class SpanPairDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing sentences with a target span and a label.
    Expected format for each input line: {"sentence": "text", "start": "int", "end": int, "label": "text"}

    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``
        span: ``SpanField``
        span_text: ``TextField``
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``
    spacy_languages: ``Dict[str, str]``, optional
        Tokenizer to use to split the sentence into words. Defaults to ``WordTokenizer()``.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 spacy_languages: Dict[str, str],
                 skip_label_indexing=False) -> None:
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True)
        self._token_indexers = token_indexers
        self._tokenizer = WhitespaceTokenizer()
        self.nlp_dict = {k: spacy.load(v) for k, v in spacy_languages.items()}
        self.skip_label_indexing = skip_label_indexing

    def clean_text(self, nlp, text):
        doc = nlp(text.replace("-", " "))
        words = [token.text for token in doc if not token.is_space]
        return ' '.join(words)

    def parse_with_offset(self, nlp, sentence, span):

        sentence = self.clean_text(nlp, sentence)
        span = self.clean_text(nlp, span)

        doc = nlp(sentence)
        span_doc = nlp(span)

        score = np.zeros(len(doc) - len(span_doc))
        for i, text_window in enumerate(windowed(doc, len(span_doc))):
            if i < len(doc) - len(span_doc):
                for t, s in zip(text_window, [t for t in span_doc]):
                    score[i] += distance(t.lower_, s.lower_)

        start_idx = score.argmin()
        start_token = doc[start_idx]
        offset = start_token.idx
        end_offset = offset + len(span)
        end_tokens = [t for t in doc if t.idx <= end_offset <= t.idx + len(t.text)]
        end_token = end_tokens[0]
        return {
            'sentence': sentence,
            'span': span,
            "offsets": [[t.idx, t.idx + len(t.text)] for t in doc if start_token.i <= t.i <= end_token.i],
            "start": start_token.i,
            "end": end_token.i,
        }

    def _read(self, file_path) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            example_iter = (json.loads(line) for line in data_file if line)
            filtered_example_iter = (
                example for example in example_iter
            )
            for example in self.shard_iterable(filtered_example_iter):
                try:
                    yield self.text_to_instance(example)
                except IndexError:
                    logger.warning(f"Parsing failed: {example}")
                except TypeError:
                    logger.warning(f"Parsing failed: {example}")

    def text_to_instance(self, example: dict) -> Instance:
        lang = example['lang'].lower()
        nlp = self.nlp_dict.get(lang, self.nlp_dict['en'])

        fields = {}
        for k in ['sentence1', 'sentence2']:
            if k in example:
                example.update({
                    k: self.clean_text(nlp, example[k]),
                })

                tokenized_sentence = self._tokenizer.tokenize(example[k])
                fields[k] = TextField(tokenized_sentence)

        span1 = self.parse_with_offset(nlp, example['sentence1'], example['MWE1'])
        span1_field = ListField([SpanField(span1['start'],
                                          min(span1['end'], fields['sentence1'].sequence_length() - 1),
                                          fields['sentence1'])])
        span1_text = ' '.join(span1['sentence'].split()[span1['start']:span1['end'] + 1])
        tokenized_span1 = self._tokenizer.tokenize(span1_text)
        span1_text_field = TextField(tokenized_span1)

        if example['MWE2'] in ['None', '']:
            start = span1['start']
            end = len(example['sentence2'].split()) - (len(span1['sentence'].split()) - span1['end'])
            example['MWE2'] = ' '.join(example['sentence2'].split()[start:end+1])

        span2 = self.parse_with_offset(nlp, example['sentence2'], example['MWE2'])
        span2_field = ListField([SpanField(span2['start'],
                                           min(span2['end'], fields['sentence2'].sequence_length() - 1),
                                           fields['sentence2'])])
        span2_text = ' '.join(span2['sentence'].split()[span2['start']:span2['end'] + 1])
        tokenized_span2 = self._tokenizer.tokenize(span2_text)
        span2_text_field = TextField(tokenized_span2)

        fields.update({
            'span1': span1_field,
            'span1_text': span1_text_field,
            'span2': span2_field,
            'span2_text': span2_text_field
        })

        label = example.get('label')
        if label is not None:
            fields['label'] = FloatField(label)

        fields['metadata'] = MetadataField(example)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance['sentence1'].token_indexers = self._token_indexers
        instance["span1_text"].token_indexers = self._token_indexers
        instance['sentence2'].token_indexers = self._token_indexers
        instance["span2_text"].token_indexers = self._token_indexers
