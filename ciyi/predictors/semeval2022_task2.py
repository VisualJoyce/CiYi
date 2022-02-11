import logging

from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from ciyi import SpanClassifierPredictor

logger = logging.getLogger(__name__)


@Predictor.register('semeval-2022_task02_idiomacity_subtask_a')
class SemEval2022Task2SubtaskAPredictor(SpanClassifierPredictor):
    """"Predictor wrapper for the SentenceSpanClassificationModel"""

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return f"{outputs['metadata']['ID']},{outputs['metadata']['Language']},{outputs['metadata']['Setting']},{outputs['label']}\n"


@Predictor.register('semeval-2022_task02_idiomacity_subtask_b')
class SemEval2022Task2SubtaskBPredictor(SpanClassifierPredictor):
    """"Predictor wrapper for the SentenceSpanClassificationModel"""

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return f"{outputs['metadata']['ID']},{outputs['metadata']['Language']},{outputs['metadata']['Setting']},{outputs['sim']}\n"
