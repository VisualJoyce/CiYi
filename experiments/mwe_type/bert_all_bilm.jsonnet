local MODEL_NAME = std.extVar("MODEL_NAME");
local ANNOTATION_DIR = std.extVar("ANNOTATION_DIR");

{
  "dataset_reader": {
    "type": "sequence",
    "token_indexers": {
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "model_name": MODEL_NAME
      }
    }
  },
  "train_data_path": ANNOTATION_DIR + "/train.jsonl",
  "validation_data_path": ANNOTATION_DIR + "/val.jsonl",
  "model": {
    "type": "sequence_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer_mismatched",
          "model_name": MODEL_NAME,
          "last_layer_only": false,
          "train_parameters": false
        }
      }
    },
    "sentence_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 768,
      "hidden_size": 768,
      "num_layers": 1
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
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      "type": "adagrad"
    }
  }
}