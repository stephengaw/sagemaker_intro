# SageMaker Intro

## Setup

The following environment variable are required to be set, identifying the AWS account you want to use:

```
export DEMO_AWS_REGION="{The region you are using}"
export DEMO_AWS_ACCOUNT={Account number}
export DEMO_AWS_PROFILE_NAME="{Name of profile to use in the .aws/credentials}"
```

## Running

1. Create data by running `make_dummy_data.py`
2. Sequentially go through the subfolders 01 to 06 to see how the ML script builds up and is run on a local machine and then SageMaker
