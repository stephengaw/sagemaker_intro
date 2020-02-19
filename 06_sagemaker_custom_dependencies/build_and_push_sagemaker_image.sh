#!/usr/bin/env bash
# Script to build and push a docker image to ECR

# --- SET VARIABLES ---
image_name=sagemaker-sklearn-expanded
fullname="${DEMO_AWS_ACCOUNT}.dkr.ecr.${DEMO_AWS_REGION}.amazonaws.com/${image_name}:latest"

# --- RUN COMMANDS ---
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image_name}" --region eu-west-1 --profile sandbox-admin

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image_name}" --region eu-west-1 --profile sandbox-admin
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${DEMO_AWS_REGION} --profile ${DEMO_AWS_ACCOUNT} --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker Tensorflow image
# NOTE: We need both login commands
$(aws ecr get-login --registry-ids 141502667606 --region ${DEMO_AWS_REGION} --profile ${DEMO_AWS_ACCOUNT} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${image_name} .
docker tag ${image_name} ${fullname}

docker push ${fullname}
