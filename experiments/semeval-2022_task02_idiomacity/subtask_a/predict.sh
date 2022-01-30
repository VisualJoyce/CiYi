#!/bin/bash
# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
#WORK_DIR=$(readlink -f .)
#ANNOTATION_DIR=$PWD/data/annotations/Math23K/
MODEL_NAME=$1
PHASE_NAME=$2
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib

declare -a datasets=(ZeroShot OneShot)
declare -a splits=(eval test)
#declare -a phase=(practice evaludation)

for d in "${datasets[@]}"
do
  for s in "${splits[@]}"
  do
    TRANSFORMER_LAYER=12 ANNOTATION_DIR=data/annotations/semeval-2022_task02_idiomacity/subtask_a/"$PHASE_NAME"/"$d" \
    allennlp predict \
    data/output/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/"$d"/finetune/"$MODEL_NAME"/model.tar.gz \
    data/annotations/semeval-2022_task02_idiomacity/subtask_a/"$d"/"$s".jsonl \
    --predictor semeval-2022_task02_idiomacity_subtask_a \
    --output-file data/output/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/"$d"/finetune/"$MODEL_NAME"/"$s"_predict.csv \
    --include-package ciyi --cuda-device 0
  done
done
