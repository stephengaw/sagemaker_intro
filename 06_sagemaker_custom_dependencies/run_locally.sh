#!/usr/bin/env bash
# Bash script to automate the running of dummy_ml_script_with_args.py on a local machine
# specify the right python environment: `conda activate sagemaker_intro`

# run ML script
python ml_script_with_dependancies.py --train="../data" --model-dir="../models" --penalty="l1" -C="1.0"
