#!/bin/bash
WORK_DIR=$(dirname $(readlink -f $0))
DATA_DIR=$1
PRETRAINED_MODEL_PATH=$2

OUTPUT_DIR=$DATA_DIR/output
SUBMISSION_DIR=$DATA_DIR/submissions
ANNOTATION_DIR=$DATA_DIR/annotations/semeval-2022_task02_idiomacity/subtask_a
CONFIGURATION_DIR=experiments/semeval-2022_task02_idiomacity/subtask_a

#declare -a models=(bert-base-multilingual-cased xlm-roberta-base xlm-roberta-large)
declare -a models=(bert-base-multilingual-cased xlm-roberta-base)
#declare -a models=(xlm-roberta-base)
declare -a layers=(24 12 8 4)
#declare -a layers=(12)
declare -a span_extractor_types=("endpoint" "self_attentive" "max_pooling")
#declare -a span_extractor_types=("max_pooling")
declare -a combinations=("x,y" "x,y,xy" "x,y,x-y" "x,y,xy,x-y")
#declare -a combinations=("x,y")
declare -a datasets=(ZeroShot OneShot)
declare -a splits=(eval test)

#declare -a phase=(practice evaluation)
PHASE_NAME=evaluation

cd "$WORK_DIR" || exit

function inference() {
  # bash predict.sh "$m" evaluation "$c" "$l" "$ANNOTATION_DIR" "$OUTPUT_DIR"
  for d in "${datasets[@]}"; do
    for s in "${splits[@]}"; do
      MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/"$d"/finetune/"$1"
      PREDICT_OUTPUT="${MODEL_PATH}"/"$s"_predict.csv
      if [ ! -f "$PREDICT_OUTPUT" ]; then
        TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR="$ANNOTATION_DIR"/"$PHASE_NAME"/"$d" \
          allennlp predict \
          "${MODEL_PATH}"/model.tar.gz \
          "${ANNOTATION_DIR}"/"$PHASE_NAME"/"$d"/"$s".jsonl \
          --predictor semeval-2022_task02_idiomacity_subtask_a \
          --output-file "$PREDICT_OUTPUT" \
          --include-package ciyi --cuda-device 0
      fi
    done
  done
}

function submission() {
  setting="${SUBMISSION_DIR}/$1"
  mkdir -p "$setting"
  echo "ID,Language,Setting,Label" >"$setting"/task2_subtaska.csv
  cat "${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/evaluation/ZeroShot/finetune/"$1"/test_predict.csv >>"$setting"/task2_subtaska.csv
  cat "${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/evaluation/OneShot/finetune/"$1"/test_predict.csv >>"$setting"/task2_subtaska.csv
}

function train_endpoint() {
  for c in "${combinations[@]}"; do
    setting="endpoint/$m/$c/$l"
    echo "$setting"

    if [ -z "$PRETRAINED_MODEL_PATH" ]; then
      mp=$m
    else
      mp=$PRETRAINED_MODEL_PATH/$m
    fi

    ZERO_SHOT_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/ZeroShot/finetune/"$setting"
    if [ ! -f "$ZERO_SHOT_MODEL_PATH"/model.tar.gz ]; then
      rm -r "$ZERO_SHOT_MODEL_PATH"
      # bash train.sh "$m" evaluation "$c" "$l" "$ANNOTATION_DIR" "$OUTPUT_DIR"
      TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/$PHASE_NAME/ZeroShot \
        MODEL_NAME=$mp SPAN_EXTRACTOR_TYPE=endpoint ENDPOINT_SPAN_EXTRACTOR_COMBINATION=$c \
        allennlp train ${CONFIGURATION_DIR}/zero_shot_finetune.jsonnet \
        -s "${ZERO_SHOT_MODEL_PATH}" \
        --include-package ciyi
    fi

    ONE_SHOT_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/OneShot/finetune/"$setting"
    if [ ! -f "$ONE_SHOT_MODEL_PATH"/model.tar.gz ]; then
      rm -r "$ONE_SHOT_MODEL_PATH"
      TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/$PHASE_NAME/OneShot \
        MODEL_NAME=$mp SPAN_EXTRACTOR_TYPE=endpoint ENDPOINT_SPAN_EXTRACTOR_COMBINATION=$c \
        allennlp train ${CONFIGURATION_DIR}/one_shot_finetune.jsonnet \
        -s "${ONE_SHOT_MODEL_PATH}" \
        --include-package ciyi
    fi

    inference "$setting"
    submission "$setting"
  done
}

function train_others() {
  setting="$1/$m/$l"
  echo "$setting"

  if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    mp=$m
  else
    mp=$PRETRAINED_MODEL_PATH/$m
  fi

  # bash train.sh "$m" evaluation "$c" "$l" "$ANNOTATION_DIR" "$OUTPUT_DIR"
  ZERO_SHOT_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/ZeroShot/finetune/"$setting"
  if [ ! -f "$ZERO_SHOT_MODEL_PATH"/model.tar.gz ]; then
    rm -r "$ZERO_SHOT_MODEL_PATH"
    TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/$PHASE_NAME/ZeroShot \
      MODEL_NAME=$mp SPAN_EXTRACTOR_TYPE=$s \
      allennlp train ${CONFIGURATION_DIR}/zero_shot_finetune.jsonnet \
      -s "${ZERO_SHOT_MODEL_PATH}" \
      --include-package ciyi
  fi

  ONE_SHOT_MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskA/"$PHASE_NAME"/OneShot/finetune/"$setting"
  if [ ! -f "$ONE_SHOT_MODEL_PATH"/model.tar.gz ]; then
    rm -r "$ONE_SHOT_MODEL_PATH"
    TOKENIZERS_PARALLELISM=false TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/$PHASE_NAME/OneShot \
      MODEL_NAME=$mp SPAN_EXTRACTOR_TYPE=$s \
      allennlp train ${CONFIGURATION_DIR}/one_shot_finetune.jsonnet \
      -s "${ONE_SHOT_MODEL_PATH}" \
      --include-package ciyi
  fi

  inference "$setting"
  submission "$setting"
}

for m in "${models[@]}"; do
  for l in "${layers[@]}"; do
    if [[ "$m" != 'xlm-roberta-large' && "$l" -gt 12 ]]; then
      continue
    fi
    for s in "${span_extractor_types[@]}"; do
      if [ "$s" == "endpoint" ]; then
        train_endpoint
      else
        train_others "$s"
      fi
    done
  done
done
