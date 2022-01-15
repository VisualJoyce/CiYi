import json
import logging
from typing import Dict
import pandas as pda

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
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

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = None) -> None:
        super().__init__()
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer or SpacyTokenizer()

    @overrides
    def _read(self, file_path):
        df = pda.read_csv(file_path)

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                curr_example_json = json.loads(line)
                sentence = curr_example_json['sentence']
                target_start = curr_example_json['start']
                target_end = curr_example_json['end']
                label = curr_example_json['label']
                yield self.text_to_instance(sentence, target_start, target_end, label)

    @overrides
    def text_to_instance(self, sentence: str, start: int, end: int, label: str = None) -> Instance:
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
