from typing import Dict, Optional

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from allennlp.training.metrics import SpearmanCorrelation
from overrides import overrides


@Model.register("span_embedder")
class SpanEmbedder(Model):
    """
    This ``Model`` performs classification of sentence (with a given span of interest) to a label.

    We embed the sentence with the text_field_embedder, and possibly encode it.
    Then we apply an extractor to get the vectors associated with the span.

    We feed this vector into a FF network for classification.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    classifier_feedforward : ``FeedForward``
    span_extractor: ``SpanExtractor``
        If provided, will combine the span into one vector
    seq2seq_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        The encoder that we will use to convert the sentence to a sequence of vectors.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            seq2vec_encoder: Seq2VecEncoder = None,
            span_extractor: Optional[SpanExtractor] = None,
            namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._span_extractor = span_extractor
        self._namespace = namespace

        self._correlation = SpearmanCorrelation()
        self._loss = nn.MSELoss()
        initializer(self)

    def encode(self, sentence: Dict[str, torch.LongTensor]):
        embedded_text = self._text_field_embedder(sentence)
        sentence_mask = util.get_text_field_mask(sentence)

        # Encode the sequence
        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, sentence_mask)

        return self._seq2vec_encoder(embedded_text, mask=sentence_mask)

    def forward(self,  # type: ignore
                sentence1: Dict[str, torch.LongTensor],
                span1: torch.LongTensor,
                span1_text: Dict[str, torch.LongTensor],
                sentence2: Dict[str, torch.LongTensor],
                span2: torch.LongTensor,
                span2_text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata=None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentence : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        span : torch.LongTensor, required
            The span field
        span_text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
            :param label:
            :param sentence1:
            :param sentence2:
            :param metadata:
        """
        embedded_text1 = self.encode(sentence1)
        embedded_text2 = self.encode(sentence2)
        output_dict = {'sentence_embedding': embedded_text1, "metadata": metadata}

        # Extract the span: shape = (batch_size, num_spans, feed_forward.input_dim())
        embedded_span1 = self._span_extractor(embedded_text1, span1)
        if len(embedded_text1.shape) == 3:
            embedded_span1 = embedded_span1.squeeze(1)

        embedded_span2 = self._span_extractor(embedded_text2, span2)
        if len(embedded_text1.shape) == 3:
            embedded_span2 = embedded_span2.squeeze(1)

        sim = torch.cosine_similarity(embedded_span1, embedded_span2)
        output_dict["sim"] = sim
        if label is not None:
            self._correlation(sim, label)
            loss = self._loss(sim, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a ``"label"`` key to the result.
        """
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "spearman_correlation": self._correlation.get_metric(reset),
        }
        return metrics
