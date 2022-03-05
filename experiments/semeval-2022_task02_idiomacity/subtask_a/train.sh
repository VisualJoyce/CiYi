#!/bin/bash
# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
#WORK_DIR=$(readlink -f .)
#ANNOTATION_DIR=$PWD/data/annotations/Math23K/
MODEL_NAME=$1
PHASE_NAME=$2
SPAN_EXTRACTOR_TYPE=$3
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib

TRANSFORMER_LAYER=12 ANNOTATION_DIR=data/annotations/semeval-2022_task02_idiomacity/subtask_a/$PHASE_NAME/ZeroShot \
  MODEL_NAME=$MODEL_NAME SPAN_EXTRACTOR_TYPE=$SPAN_EXTRACTOR_TYPE\
  allennlp train experiments/semeval-2022_task02_idiomacity/subtask_a/zero_shot_finetune.jsonnet \
  -s data/output/semeval-2022_task02_idiomacity/SubTaskA/evaluation/ZeroShot/finetune/"$MODEL_NAME"/"$SPAN_EXTRACTOR_TYPE" \
  --include-package ciyi

TRANSFORMER_LAYER=12 ANNOTATION_DIR=data/annotations/semeval-2022_task02_idiomacity/subtask_a/$PHASE_NAME/OneShot \
  MODEL_NAME=$MODEL_NAME SPAN_EXTRACTOR_TYPE=$SPAN_EXTRACTOR_TYPE \
  allennlp train experiments/semeval-2022_task02_idiomacity/subtask_a/one_shot_finetune.jsonnet \
  -s data/output/semeval-2022_task02_idiomacity/SubTaskA/evaluation/OneShot/finetune/"$MODEL_NAME"/"$SPAN_EXTRACTOR_TYPE" \
  --include-package ciyi
