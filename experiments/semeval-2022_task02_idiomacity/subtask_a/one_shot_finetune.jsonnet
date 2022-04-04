local TRANSFORMER_LAYER = std.extVar("TRANSFORMER_LAYER");
local MODEL_NAME = std.extVar("MODEL_NAME");
local ANNOTATION_DIR = std.extVar("ANNOTATION_DIR");
local SPAN_EXTRACTOR_TYPE = std.extVar("SPAN_EXTRACTOR_TYPE");
local HIDDEN_DIM = if std.member(MODEL_NAME, 'xlm-roberta-large') then 1024 else 768;
local MAX_TOKENS = if std.member(MODEL_NAME, 'xlm-roberta-large') then 800 else 800;
local NUM_GRADIENT_ACCUMULATION_STEPS = if std.member(MODEL_NAME, 'xlm-roberta-large') then 8 else 8;


local SPAN_EXTRACTOR = if SPAN_EXTRACTOR_TYPE == "endpoint" then {
      "type": "endpoint",
      "combination": std.strReplace(std.extVar("ENDPOINT_SPAN_EXTRACTOR_COMBINATION"), "xy", "x*y"),
      "input_dim": HIDDEN_DIM
    } else {
      "type": SPAN_EXTRACTOR_TYPE ,
      "input_dim": HIDDEN_DIM
    };


{
  "dataset_reader": {
    "type": "span",
    "token_indexers": {
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "max_length": std.min(512, MAX_TOKENS),
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
          "max_length": std.min(512, MAX_TOKENS),
          "last_layer_only": false,
          "transformer_layer": TRANSFORMER_LAYER,
          "train_parameters": true
        }
      }
    },
    "dropout": 0.5,
    "num_labels" : 2,
    "span_extractor": SPAN_EXTRACTOR
  },
  "data_loader": {
    "num_workers": 8,
    "batch_sampler": {
      "type": "max_tokens_sampler",
      "max_tokens": MAX_TOKENS
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
    "patience" : 10,
    "num_gradient_accumulation_steps": NUM_GRADIENT_ACCUMULATION_STEPS,
    "cuda_device": 0,
    "validation_metric": "+f1"
  }
}
