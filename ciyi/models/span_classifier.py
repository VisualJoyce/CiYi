from typing import Dict, Optional

import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides


@Model.register("span_classifier")
class SpanClassifier(Model):
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
            # seq2vec_encoder: Seq2VecEncoder,
            span_extractor: Optional[SpanExtractor] = None,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            num_labels: int = None,
            label_namespace: str = "labels",
            namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._span_extractor = span_extractor
        # self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._span_extractor.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._f1 = F1Measure(positive_label=1)
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                span: torch.LongTensor,
                span_text: Dict[str, torch.LongTensor],
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
        """
        embedded_text = self._text_field_embedder(sentence)

        # Encode the sequence
        if self._seq2seq_encoder:
            sentence_mask = util.get_text_field_mask(sentence)
            embedded_text = self._seq2seq_encoder(embedded_text, sentence_mask)

        # Extract the span: shape = (batch_size, num_spans, feed_forward.input_dim())
        embedded_text = self._span_extractor(embedded_text, span)
        if len(embedded_text.shape) == 3:
            embedded_text = embedded_text.squeeze(0)

        # span_mask = util.get_text_field_mask(span_text)
        # embedded_text = self._seq2vec_encoder(embedded_text, mask=span_mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs, "metadata": metadata}
        # output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a ``"label"`` key to the result.
        """
        # output_dict = super(SentenceSpanClassificationModel, self).decode(output_dict)
        label_probs = torch.nn.functional.softmax(output_dict['logits'], dim=-1)
        output_dict['label_probs'] = label_probs
        predictions = label_probs.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)

        # Single instance
        if np.isscalar(argmax_indices):
            argmax_indices = [argmax_indices]

        if not output_dict['metadata'][0]['skip_indexing']:
            labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        else:
            labels = argmax_indices
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
        }
        metrics.update(self._f1.get_metric(reset))
        return metrics
