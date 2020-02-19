# Region and Account for AWS base image
ARG REGION=eu-west-1
ARG ACCOUNT=141502667606

# SageMaker Tensorflow image
FROM $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3

RUN pip install tabulate

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our utils module
COPY utils.py /opt/ml/code/utils.py
COPY ml_script_with_dependancies.py /opt/ml/code/ml_script_with_dependancies.py

# this environment variable is used by the SageMaker Tensorflow container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code