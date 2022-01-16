import json
import logging
from typing import Dict

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

    @staticmethod
    def parse_start_end(doc, span_doc):
        possessive_form1 = {
            "one",
            "someone",
            "anyone"
        }

        possessive_form2 = {'my', 'your', 'his', 'her', 'our', 'their'}

        start, end = None, None
        for i, t in enumerate(doc):
            j = i
            trues = []
            for tt in span_doc:
                if t.is_punct:
                    j += 1
                elif tt.is_punct or tt.text == "'s":
                    continue
                else:
                    checks = [
                        t.text.lower() == tt.text.lower(),
                        t.text.lower() in possessive_form2 and tt.text.lower() in possessive_form1,
                        t.lemma_.lower() == tt.lemma_.lower()
                    ]
                    if any(checks):
                        trues.append(True)
                    else:
                        trues.append(False)
                        break
                    j += 1
                t = doc[j]
            if trues and all(trues):
                start = i
                end = j - 1
                break

        if start is None or end is None:
            raise IndexError

        return start, end

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
                    yield self.text_to_instance(curr_example_json)
                except IndexError:
                    logger.warning(f"Parsing failed: {line}")

    @overrides
    def text_to_instance(self, example: dict) -> Instance:
        sentence = example['sentence']

        doc = self._tokenizer.spacy(sentence)
        if all([k in example for k in ('start', 'end')]):
            start = example['start']
            end = example['end']
        else:
            span_doc = self._tokenizer.spacy(sentence)
            start, end = self.parse_start_end(doc, span_doc)
        tokenized_sentence = self._tokenizer._sanitize(doc)

        label = example['label']
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        span_field = SpanField(start, end, sentence_field)

        span_text = ' '.join(sentence.split()[start:end + 1])
        tokenized_span = self._tokenizer.tokenize(span_text)
        span_text_field = TextField(tokenized_span, self._token_indexers)

        fields = {'sentence': sentence_field, 'span': span_field, 'span_text': span_text_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
