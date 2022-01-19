import json
import logging
from typing import Dict

import numpy as np
import spacy
from more_itertools import windowed
from Levenshtein import distance
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("span")
class SpanDatasetReader(DatasetReader):
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
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into words. Defaults to ``WordTokenizer()``.
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer], spacy_languages: Dict[str, str]) -> None:
        super().__init__()
        self._token_indexers = token_indexers
        self._tokenizer = WhitespaceTokenizer()
        self.nlp_dict = {k: spacy.load(v) for k, v in spacy_languages.items()}

    @staticmethod
    def parse_with_offset(nlp, sentence, span):

        doc = nlp(sentence.replace("-", " "))
        span_doc = nlp(span.replace("-", " "))

        sentence_words = [token.text for token in doc if not token.is_space]
        sentence = ' '.join(sentence_words)

        target_paraphrase_words = [token.text for token in span_doc if not token.is_space]
        target_paraphrase = ' '.join(target_paraphrase_words)

        doc = nlp(sentence)

        score = np.zeros(len(sentence_words))
        for i, text_window in enumerate(windowed(sentence_words, len(target_paraphrase_words))):
            for t, s in zip(text_window, target_paraphrase_words):
                score[i] += distance(t.lower(), s.lower())
        start_idx = score.argmin()
        start_token = doc[start_idx]
        offset = start_token.idx
        end_offset = offset + len(target_paraphrase)
        end_tokens = [t for t in doc if t.idx <= end_offset <= t.idx + len(t.text)]
        end_token = end_tokens[0]
        d = {
            'sentence': sentence,
            'span': span,
            "offsets": [[t.idx, t.idx + len(t.text)] for t in doc if start_token.i <= t.i <= end_token.i],
            "start": start_token.i,
            "end": end_token.i,
        }

        return d

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                curr_example_json = json.loads(line)
                try:
                    if not all([k in curr_example_json for k in ('start', 'end')]):
                        lang = curr_example_json['lang'].lower()
                        nlp = self.nlp_dict.get(lang)
                        curr_example_json.update(self.parse_with_offset(nlp,
                                                                        curr_example_json['sentence'],
                                                                        curr_example_json['span']))
                    yield self.text_to_instance(curr_example_json)
                except IndexError:
                    logger.warning(f"Parsing failed: {line}")

    @overrides
    def text_to_instance(self, example: dict) -> Instance:
        sentence = example['sentence']
        start = example['start']
        end = example['end']
        label = example['label']

        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        span_field = SpanField(start, end, sentence_field)

        span_text = ' '.join(sentence.split()[start:end + 1])
        tokenized_span = self._tokenizer.tokenize(span_text)
        span_text_field = TextField(tokenized_span, self._token_indexers)

        fields = {'sentence': sentence_field, 'span': span_field, 'span_text': span_text_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
