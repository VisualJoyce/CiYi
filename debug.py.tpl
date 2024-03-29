import json
import shutil
import sys

from allennlp.commands import main

config_file = "experiments/semeval-2022_task02_idiomacity/subtask_a/zero_shot_finetune.jsonnet"

# Use overrides to train on CPU.
# overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "data/output/semeval-2022_task02_idiomacity/SubTaskA/zero_shot/finetune"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "ciyi",
    # "-o", overrides,
]

main()