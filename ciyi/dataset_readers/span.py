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
        self._tokenizer = tokenizer or SpacyTokenizer(split_on_spaces=True)

    # @staticmethod
    # def parse_with_offset(doc, span_doc):
    #
    #     sentence_words = [token.text for token in doc if not token.is_space]
    #     sentence = ' '.join(sentence_words)
    #
    #     target_paraphrase_words = [token.text for token in span_doc if not token.is_space]
    #     target_paraphrase = ' '.join(target_paraphrase_words)
    #
    #     doc = nlp(sentence)
    #
    #     count = sentence.lower().count(target_paraphrase.lower())
    #     if count == 1:
    #         offset = sentence.lower().index(target_paraphrase.lower())
    #     elif count > 1:
    #         sentence_words_lower = [w.lower() for w in sentence_words]
    #         p_count = sentence_words_lower.count(target_paraphrase_words[0].lower())
    #         if p_count == 1:
    #             left = sentence_words_lower.index(target_paraphrase_words[0].lower())
    #             offset = doc[left].idx
    #         else:
    #             return
    #     else:
    #         return
    #
    #     start_tokens = [t for t in doc if t.idx == offset]
    #     if start_tokens:
    #         start_token = start_tokens[0]
    #         end_offset = offset + len(target_paraphrase)
    #         end_tokens = [t for t in doc if t.idx + len(t.text) == end_offset]
    #         if end_tokens:
    #             end_token = end_tokens[0]
    #             d = {
    #                 'sentence': sentence,
    #                 'span': span,
    #                 "offsets": [[t.idx, t.idx + len(t.text)] for t in doc if start_token.i <= t.i <= end_token.i],
    #             }
    #             return d

    @staticmethod
    def parse_start_end(doc, span_doc):

        # context = sentence1.replace("-", " ").split()
        # span = sentence2.replace("-", " ").split()
        # cost = np.zeros((len(context), len(span)))
        # for i, aw in enumerate(context):
        #     for j, dw in enumerate(span):
        #         cost[i, j] = -jaro_winkler(aw, dw)
        #
        # row_ind, col_ind = linear_sum_assignment(cost)
        # print(row_ind, col_ind, context[row_ind:col_ind])
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
                        t.lemma_.lower() == tt.lemma_.lower(),
                        t.lemma_.lower() in tt.lemma_.lower(),
                        tt.lemma_.lower() in t.lemma_.lower(),
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

        if all([k in example for k in ('start', 'end')]):
            doc = self._tokenizer.spacy(sentence)
            start = example['start']
            end = example['end']
        else:
            doc = self._tokenizer.spacy(sentence.replace("-", " "))
            span_doc = self._tokenizer.spacy(sentence.replace("-", " "))
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
