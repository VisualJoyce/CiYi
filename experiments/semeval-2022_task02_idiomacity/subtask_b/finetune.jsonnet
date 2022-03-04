local TRANSFORMER_LAYER = std.extVar("TRANSFORMER_LAYER");
local MODEL_NAME = std.extVar("MODEL_NAME");
local ANNOTATION_DIR = std.extVar("ANNOTATION_DIR");

{
  "dataset_reader": {
    "type": "sentence_pair",
    "token_indexers": {
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "max_length": 512,
        "model_name": MODEL_NAME
      }
    },
    "spacy_languages": {
        "en": "en_core_web_sm",
        "pt": "pt_core_news_sm",
    },
    "skip_label_indexing": true
  },
  "train_data_path": ANNOTATION_DIR + "/train.jsonl",
  "validation_data_path": ANNOTATION_DIR + "/validation.jsonl",
  "model": {
    "type": "from_archive",
    "archive_file": "data/output/semeval-2022_task02_idiomacity/SubTaskB/pretrain/" + MODEL_NAME + "/model.tar.gz"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 20
    }
  },
  "trainer": {
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 3e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "correct_bias": true
    },
    "learning_rate_scheduler": {
        "type": "polynomial_decay",
    },
    "grad_norm": 1.0,
    "num_epochs": 10,
    "patience" : 3,
    "num_gradient_accumulation_steps": 8,
    "cuda_device": 0,
    "validation_metric": "+spearman_correlation"
  }
}