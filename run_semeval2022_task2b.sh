#!/bin/bash
WORK_DIR=$(dirname $(readlink -f $0))
DATA_DIR=$1
PRETRAINED_MODEL_PATH=$2

OUTPUT_DIR=$DATA_DIR/output
SUBMISSION_DIR=$DATA_DIR/submissions
ANNOTATION_DIR=$DATA_DIR/annotations/semeval-2022_task02_idiomacity/subtask_b
CONFIGURATION_DIR=experiments/semeval-2022_task02_idiomacity/subtask_b

declare -a models=(bert-base-multilingual-cased xlm-roberta-base xlm-roberta-large)
#declare -a models=(bert-base-multilingual-cased)
#declare -a models=(xlm-roberta-base xlm-roberta-large)
declare -a seq2vec_encoder_types=("boe" "cls_pooler")
declare -a splits=(finetune_train finetune_validation test)

cd "$WORK_DIR" || exit

for m in "${models[@]}"; do
  if [ "$m" == 'xlm-roberta-large' ]; then
    l=24
  else
    l=12
  fi

  for c in "${seq2vec_encoder_types[@]}"; do
    setting="${SUBMISSION_DIR}/$m-$l-$c"
    echo "$setting"

    if [ -z "$PRETRAINED_MODEL_PATH" ]; then
      mp=$m
    else
      mp=$PRETRAINED_MODEL_PATH/$m
    fi

    PRETRAIN_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/pretrain/"$m"/"$l"/"$c"
    FINETUNE_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/finetune/"$m"/"$l"/"$c"

    TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/pretrain \
      MODEL_NAME=$mp SEQ2VEC_ENCODER_TYPE=$c \
      allennlp train ${CONFIGURATION_DIR}/pretrain.jsonnet \
      -s "${PRETRAIN_MODEL_PATH}" \
      --include-package ciyi

    for s in "${splits[@]}"; do
      allennlp predict \
        "${PRETRAIN_MODEL_PATH}"/model.tar.gz \
        "${ANNOTATION_DIR}"/predict/"$s".jsonl \
        --predictor semeval-2022_task02_idiomacity_subtask_b \
        --output-file "${PRETRAIN_MODEL_PATH}"/"$s"_predict.csv \
        --include-package ciyi --cuda-device 0
    done

    python3 ${CONFIGURATION_DIR}/update_data.py \
      --annotation_location "${ANNOTATION_DIR}"/finetune \
      --prediction_location "${PRETRAIN_MODEL_PATH}"

    TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/finetune \
      MODEL_NAME=$mp SEQ2VEC_ENCODER_TYPE=$c \
      allennlp train ${CONFIGURATION_DIR}/finetune.jsonnet \
      -s "${FINETUNE_MODEL_PATH}" \
      --include-package ciyi

    allennlp predict \
      "${FINETUNE_MODEL_PATH}"/model.tar.gz \
      "${ANNOTATION_DIR}"/finetune/test.jsonl \
      --predictor semeval-2022_task02_idiomacity_subtask_b \
      --output-file "${FINETUNE_MODEL_PATH}"/test_predict.csv \
      --include-package ciyi --cuda-device 0

    mkdir -p "$setting"
    echo "ID,Language,Setting,Sim" >"$setting"/task2_subtaskb.csv
    cat "${PRETRAIN_MODEL_PATH}"/test_predict.csv >>"$setting"/task2_subtaskb.csv
    cat "${FINETUNE_MODEL_PATH}"/test_predict.csv >>"$setting"/task2_subtaskb.csv
  done
done
