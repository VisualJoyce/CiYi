local TRANSFORMER_LAYER = std.extVar("TRANSFORMER_LAYER");
local MODEL_NAME = std.extVar("MODEL_NAME");
local ANNOTATION_DIR = std.extVar("ANNOTATION_DIR");
local SPAN_EXTRACTOR_TYPE = std.extVar("SPAN_EXTRACTOR_TYPE");
local MAX_TOKENS = if MODEL_NAME == 'xlm-roberta-base' then 800 else 800;
local NUM_GRADIENT_ACCUMULATION_STEPS = if MODEL_NAME == 'xlm-roberta-base' then 8 else 8;

{
  "dataset_reader": {
    "type": "span",
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
  "validation_data_path": ANNOTATION_DIR + "/dev.jsonl",
  "model": {
    "type": "span_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer_layern_mismatched",
          "model_name": MODEL_NAME,
          "max_length": 512,
          "last_layer_only": false,
          "transformer_layer": TRANSFORMER_LAYER,
          "train_parameters": true
        }
      }
    },
    "num_labels" : 2,
    "span_extractor": {
      "type": "endpoint",
      "combination": SPAN_EXTRACTOR_TYPE,
      "input_dim": 768
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "max_tokens_sampler",
      "max_tokens": 800
    }
  },
  "trainer": {
    "optimizer": {
        "type": "huggingface_adamw",
        "lr": 5e-5,
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
    "num_gradient_accumulation_steps": NUM_GRADIENT_ACCUMULATION_STEPS,
    "cuda_device": 0,
    "validation_metric": "+f1"
  }
}