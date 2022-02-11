import logging

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

logger = logging.getLogger(__name__)


@Predictor.register('span_classifier')
class SpanClassifierPredictor(Predictor):
    """"Predictor wrapper for the SentenceSpanClassificationModel"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        try:
            instance = self._dataset_reader.text_to_instance(json_dict)
        except TypeError:
            logger.warning(f"Parsing failed: {json_dict}")
            raise TypeError
        # label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # _ = [label_dict[i] for i in range(len(label_dict))]
        return instance
