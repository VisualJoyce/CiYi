#!/bin/bash
WORK_DIR=$(readlink -f .)
PROJECT=pie_classification
DATA_DIR=${WORK_DIR}/data
PROJECT_DIR=${WORK_DIR}/experiments/$PROJECT
OUTPUT_DIR=${DATA_DIR}/output/$PROJECT
ANNOTATION_DIR=${DATA_DIR}/annotations/$PROJECT
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME=bert-base-uncased
fi

for i in {0..13}; do
  if [[ $i == 13 ]]; then
    layer="word_embeddings"
  else
    layer="$i"
  fi

  echo "$layer"

  ANNOTATION_DIR="$ANNOTATION_DIR" MODEL_NAME="$MODEL_NAME" TRANSFORMER_LAYER="$layer" allennlp train \
    "$PROJECT_DIR"/pie_classification_transformer_layer-k_bilm.jsonnet \
    -s "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm \
    --include-package ciyi

  ANNOTATION_DIR="$ANNOTATION_DIR" MODEL_NAME="$MODEL_NAME" TRANSFORMER_LAYER="$layer" allennlp evaluate \
    "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/model.tar.gz \
    "$ANNOTATION_DIR"/test.jsonl \
    --output-file "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/test_results.json \
    --include-package ciyi

  ANNOTATION_DIR="$ANNOTATION_DIR" MODEL_NAME="$MODEL_NAME" TRANSFORMER_LAYER="$layer" allennlp predict \
    "$OUTPUT_DIR"/bert_"${layer}"/bilm/model.tar.gz \
    "$ANNOTATION_DIR"/test.jsonl \
    --output-file "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/test.predictions \
    --include-package ciyi --predictor span_classifier

done
