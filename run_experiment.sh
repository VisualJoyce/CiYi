#!/bin/bash
WORK_DIR=$(readlink -f .)
DATA_DIR=${WORK_DIR}/data
PROJECT=$1
CONFIG_NAME=$2
MODEL_NAME=$3
PROJECT_DIR=${WORK_DIR}/experiments/$PROJECT
OUTPUT_DIR=${DATA_DIR}/output/$PROJECT
ANNOTATION_DIR=${DATA_DIR}/annotations/$PROJECT

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME=bert-base-uncased
fi

ANNOTATION_DIR="$ANNOTATION_DIR" MODEL_NAME="$MODEL_NAME" allennlp train \
  "$PROJECT_DIR"/"${CONFIG_NAME}".jsonnet \
  -s "$OUTPUT_DIR"/"${CONFIG_NAME}" \
  --include-package ciyi

#ANNOTATION_DIR="$ANNOTATION_DIR" MODEL_NAME="$MODEL_NAME" allennlp evaluate \
#  "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/model.tar.gz \
#  "$ANNOTATION_DIR"/test.jsonl \
#  --output-file "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/test_results.json \
#  --include-package ciyi
#
#ANNOTATION_DIR="$ANNOTATION_DIR" MODEL_NAME="$MODEL_NAME" allennlp predict \
#  "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/model.tar.gz \
#  "$ANNOTATION_DIR"/test.jsonl \
#  --output-file "$OUTPUT_DIR"/bert_layer-"${layer}"/bilm/test.predictions \
#  --include-package ciyi --predictor span_classifier
