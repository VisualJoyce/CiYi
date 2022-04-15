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
#declare -a models=(xlm-roberta-large)
declare -a seq2vec_encoder_types=("boe" "cls_pooler")
declare -a splits=(finetune_train finetune_validation test)
declare -a finetune_model_types=("sentence_embedder" "span_embedder" "sentence_span_embedder_mean" "sentence_span_embedder_concat")
declare -a span_extractor_types=("endpoint" "self_attentive" "max_pooling")
#declare -a span_extractor_types=("max_pooling")
declare -a combinations=("x,y" "x,y,xy" "x,y,x-y" "x,y,xy,x-y")

cd "$WORK_DIR" || exit
function inference() {
  FINETUNE_PREDICT_OUTPUT="${FINETUNE_MODEL_PATH}"/test_predict.csv
  if [ ! -f "$FINETUNE_PREDICT_OUTPUT" ]; then
    allennlp predict \
      "${FINETUNE_MODEL_PATH}"/model.tar.gz \
      "${ANNOTATION_DIR}"/finetune/test.jsonl \
      --predictor semeval-2022_task02_idiomacity_subtask_b \
      --output-file "${FINETUNE_PREDICT_OUTPUT}" \
      --include-package ciyi --cuda-device 0
  fi
}

function submission() {
  setting="${SUBMISSION_DIR}/$1"
  mkdir -p "$setting"
  echo "ID,Language,Setting,Sim" >"$setting"/task2_subtaskb.csv
  cat "${PRETRAIN_MODEL_PATH}"/test_predict.csv >>"$setting"/task2_subtaskb.csv
  cat "${FINETUNE_MODEL_PATH}"/test_predict.csv >>"$setting"/task2_subtaskb.csv
}

function train_endpoint() {
  for com in "${combinations[@]}"; do
    finetune_setting="$1/$com"
    echo "$finetune_setting"
    FINETUNE_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/finetune/"$finetune_setting"

    if [ -z "$PRETRAINED_MODEL_PATH" ]; then
      mp=$m
    else
      mp=$PRETRAINED_MODEL_PATH/$m
    fi

    if [ ! -f "$FINETUNE_MODEL_PATH"/model.tar.gz ]; then
      rm -r "$FINETUNE_MODEL_PATH"
      TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/finetune \
        MODEL_NAME=$mp MODEL_TYPE=$f SEQ2VEC_ENCODER_TYPE=$enc \
        SPAN_EXTRACTOR_TYPE=endpoint ENDPOINT_SPAN_EXTRACTOR_COMBINATION=$com \
        allennlp train ${CONFIGURATION_DIR}/finetune.jsonnet \
        -s "${FINETUNE_MODEL_PATH}" \
        --include-package ciyi
    fi

    FINETUNE_PREDICT_OUTPUT="${FINETUNE_MODEL_PATH}"/test_predict.csv
    if [ ! -f "$FINETUNE_PREDICT_OUTPUT" ]; then
      allennlp predict \
        "${FINETUNE_MODEL_PATH}"/model.tar.gz \
        "${ANNOTATION_DIR}"/finetune/test.jsonl \
        --predictor semeval-2022_task02_idiomacity_subtask_b \
        --output-file "${FINETUNE_PREDICT_OUTPUT}" \
        --include-package ciyi --cuda-device 0
    fi

    inference "$finetune_setting"
    submission "$finetune_setting"
  done
}

function train_others() {
  finetune_setting="$1/$com"
  echo "$finetune_setting"

  FINETUNE_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/finetune/"$finetune_setting"

  if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    mp=$m
  else
    mp=$PRETRAINED_MODEL_PATH/$m
  fi

  if [ ! -f "$FINETUNE_MODEL_PATH"/model.tar.gz ]; then
    rm -r "$FINETUNE_MODEL_PATH"
    TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/finetune \
      MODEL_NAME=$mp MODEL_TYPE=$f SEQ2VEC_ENCODER_TYPE=$enc \
      SPAN_EXTRACTOR_TYPE=$s \
      allennlp train ${CONFIGURATION_DIR}/finetune.jsonnet \
      -s "${FINETUNE_MODEL_PATH}" \
      --include-package ciyi
  fi

  inference "$finetune_setting"
  submission "$finetune_setting"
}

for m in "${models[@]}"; do
  if [ "$m" == 'xlm-roberta-large' ]; then
    l=24
  else
    l=12
  fi

  for enc in "${seq2vec_encoder_types[@]}"; do
    pretrain_setting="$m"/"$l"/"$enc"
    echo "$pretrain_setting"

    if [ -z "$PRETRAINED_MODEL_PATH" ]; then
      mp=$m
    else
      mp=$PRETRAINED_MODEL_PATH/$m
    fi

    PRETRAIN_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/pretrain/"$pretrain_setting"

    if [ ! -f "$PRETRAIN_MODEL_PATH"/model.tar.gz ]; then
      rm -r "$PRETRAIN_MODEL_PATH"
      TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/pretrain \
        MODEL_NAME=$mp SEQ2VEC_ENCODER_TYPE=$enc \
        allennlp train ${CONFIGURATION_DIR}/pretrain.jsonnet \
        -s "${PRETRAIN_MODEL_PATH}" \
        --include-package ciyi
    fi

    for s in "${splits[@]}"; do
      PREDICT_OUTPUT="${PRETRAIN_MODEL_PATH}"/"$s"_predict.csv
      if [ ! -f "$PREDICT_OUTPUT" ]; then
        allennlp predict \
          "${PRETRAIN_MODEL_PATH}"/model.tar.gz \
          "${ANNOTATION_DIR}"/predict/"$s".jsonl \
          --predictor semeval-2022_task02_idiomacity_subtask_b \
          --output-file "${PREDICT_OUTPUT}" \
          --include-package ciyi --cuda-device 0
      fi
    done

    python3 ${CONFIGURATION_DIR}/update_data.py \
      --annotation_location "${ANNOTATION_DIR}"/finetune \
      --prediction_location "${PRETRAIN_MODEL_PATH}"

    for f in "${finetune_model_types[@]}"; do
      for s in "${span_extractor_types[@]}"; do
        if [ "$s" == "endpoint" ]; then
          train_endpoint "$pretrain_setting/$f/$s"
        else
          train_others "$pretrain_setting/$f/$s"
        fi
      done
    done
  done
done
