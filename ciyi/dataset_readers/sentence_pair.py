import json
import logging
from typing import Dict

import spacy
import torch
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("sentence_pair")
class SentencePairDatasetReader(DatasetReader):
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
        super().__init__()
        self._token_indexers = token_indexers
        self._tokenizer = WhitespaceTokenizer()
        self.nlp_dict = {k: spacy.load(v) for k, v in spacy_languages.items()}
        self.skip_label_indexing = skip_label_indexing

    def clean_text(self, nlp, text):
        doc = nlp(text.replace("-", " "))
        words = [token.text for token in doc if not token.is_space]
        return ' '.join(words)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                curr_example_json = json.loads(line)
                yield self.text_to_instance(curr_example_json)
                # try:
                # except IndexError:
                #     logger.warning(f"Parsing failed: {line}")
                # except TypeError:
                #     logger.warning(f"Parsing failed: {line}")

    @overrides
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
                fields[k] = TextField(tokenized_sentence, self._token_indexers)

        label = example.get('label')
        if label is not None:
            fields['label'] = TensorField(torch.FloatTensor([label]))

        fields['metadata'] = MetadataField(example)
        return Instance(fields)
