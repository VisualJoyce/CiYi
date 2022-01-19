local TRANSFORMER_LAYER = std.extVar("TRANSFORMER_LAYER");
local MODEL_NAME = std.extVar("MODEL_NAME");
local ANNOTATION_DIR = std.extVar("ANNOTATION_DIR");

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
    }
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
          "train_parameters": false
        }
      }
    },
    "feedforward": {
      "input_dim": 3072,
      "num_layers": 1,
      "hidden_dims": [
        300
      ],
      "activations": [
        "relu"
      ],
      "dropout": [
        0.2
      ]
    },
    "num_labels" : 2,
    "seq2seq_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 768,
      "hidden_size": 768,
      "num_layers": 1
    },
    "span_extractor": {
      "type": "endpoint",
      "input_dim": 1536
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 100
    }
  },
  "trainer": {
    "num_epochs": 500,
    "patience": 20,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+f1",
    "optimizer": {
      "type": "adagrad"
    }
  }
}