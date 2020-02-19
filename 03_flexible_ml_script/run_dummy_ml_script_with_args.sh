#!/usr/bin/env bash
# Bash script to automate the running of dummy_ml_script_with_args.py on a local machine

# specify the right python environment
conda activate sagemaker_intro

# run ML script
python dummy_ml_script_with_args.py --data_dir="../data" --model_dir="../models" --penalty="l1" --C="1.0"
