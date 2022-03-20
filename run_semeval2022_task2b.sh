#!/bin/bash
WORK_DIR=$(dirname $(readlink -f $0))
DATA_DIR=$1
PRETRAINED_MODEL_PATH=$2

OUTPUT_DIR=$DATA_DIR/output
SUBMISSION_DIR=$DATA_DIR/submissions
ANNOTATION_DIR=$DATA_DIR/annotations/semeval-2022_task02_idiomacity/subtask_b
CONFIGURATION_DIR=experiments/semeval-2022_task02_idiomacity/subtask_b

#declare -a models=(bert-base-multilingual-cased)
#declare -a models=(xlm-roberta-large)
declare -a models=(xlm-roberta-base)
declare -a seq2vec_encoder_types=("boe" "cls_pooler")
declare -a layers=(12)
declare -a splits=(finetune test)

cd "$WORK_DIR" || exit

for m in "${models[@]}"; do
  for l in "${layers[@]}"; do
    for c in "${seq2vec_encoder_types[@]}"; do
        setting="${SUBMISSION_DIR}/$m-$l-$c"
        echo "$setting"

        if [ -z "$PRETRAINED_MODEL_PATH" ]; then
          mp=$m
        else
          mp=$PRETRAINED_MODEL_PATH/$m
        fi

        TRANSFORMER_LAYER=$l ANNOTATION_DIR=${ANNOTATION_DIR}/pretrain \
          MODEL_NAME=$mp SEQ2VEC_ENCODER_TYPE=$c \
          allennlp train ${CONFIGURATION_DIR}/pretrain.jsonnet \
          -s "${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/"$m"/"$l"/"$c" \
          --include-package ciyi

        for s in "${splits[@]}"; do
          MODEL_PATH="${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/"$m"/"$l"/"$c"
          allennlp predict \
            "${MODEL_PATH}"/model.tar.gz \
            "${ANNOTATION_DIR}"/predict/"$s".jsonl \
            --predictor semeval-2022_task02_idiomacity_subtask_b \
            --output-file "${MODEL_PATH}"/"$s"_predict.csv \
            --include-package ciyi --cuda-device 0
        done

        mkdir -p "$setting"
        echo "ID,Language,Setting,Sim" >"$setting"/task2_subtaskb.csv
        cat "${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/"$m"/"$l"/"$c"/test_predict.csv >>"$setting"/task2_subtaskb.csv
#        cat "${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/"$m"/"$l"/test_predict.csv >>"$setting"/task2_subtaskb.csv
        sed 's/pre_train/fine_tune/g' "${OUTPUT_DIR}"/semeval-2022_task02_idiomacity/SubTaskB/"$m"/"$l"/"$c"/test_predict.csv >>"$setting"/task2_subtaskb.csv
    done
  done
done
