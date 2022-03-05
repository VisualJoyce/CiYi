#!/bin/bash
# Copyright (c) VisualJoyce.
# Licensed under the MIT license.
#WORK_DIR=$(readlink -f .)
#ANNOTATION_DIR=$PWD/data/annotations/Math23K/
MODEL_NAME=$1
PHASE_NAME=$2
SPAN_EXTRACTOR_TYPE=$3
TRANSFORMER_LAYER=$4
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib

declare -a datasets=(ZeroShot OneShot)
declare -a splits=(eval test)
#declare -a phase=(practice evaluation)

for d in "${datasets[@]}"; do
  for s in "${splits[@]}"; do
    TRANSFORMER_LAYER=$TRANSFORMER_LAYER ANNOTATION_DIR=data/annotations/semeval-2022_task02_idiomacity/subtask_a/"$PHASE_NAME"/"$d" \
      allennlp predict \
      data/output/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/"$d"/finetune/"$MODEL_NAME"/"$SPAN_EXTRACTOR_TYPE"/"$TRANSFORMER_LAYER"/model.tar.gz \
      data/annotations/semeval-2022_task02_idiomacity/subtask_a/"$PHASE_NAME"/"$d"/"$s".jsonl \
      --predictor semeval-2022_task02_idiomacity_subtask_a \
      --output-file data/output/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/"$d"/finetune/"$MODEL_NAME"/"$SPAN_EXTRACTOR_TYPE"/"$TRANSFORMER_LAYER"/"$s"_predict.csv \
      --include-package ciyi --cuda-device 0
  done
done

if [[ $PHASE_NAME == "evaluation" ]]; then
  echo "ID,Language,Setting,Label" >task2_subtaska.csv
  cat data/output/semeval-2022_task02_idiomacity/SubTaskA/evaluation/ZeroShot/finetune/"$MODEL_NAME"/"$SPAN_EXTRACTOR_TYPE"/"$TRANSFORMER_LAYER"/test_predict.csv >>task2_subtaska.csv
  cat data/output/semeval-2022_task02_idiomacity/SubTaskA/evaluation/OneShot/finetune/"$MODEL_NAME"/"$SPAN_EXTRACTOR_TYPE"/"$TRANSFORMER_LAYER"/test_predict.csv >>task2_subtaska.csv
fi
