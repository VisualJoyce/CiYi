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
    "type": "sentence_embedder",
    "text_field_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer_layern_mismatched",
          "model_name": MODEL_NAME,
          "max_length": 512,
          "last_layer_only": false,
          "transformer_layer": TRANSFORMER_LAYER,
          "train_parameters": false
        }
      }
    },
    "seq2seq_encoder": {
      "type": "pass_through",
      "input_dim": 768,
    },
    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": 768
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 50
    }
  },
  "trainer": {
    "num_epochs": 500,
    "patience": 20,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+spearman_correlation",
    "optimizer": {
      "type": "adagrad"
    }
  }
}