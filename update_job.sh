#!/bin/bash

JOB_NAME=$1
IMAGE_URL=$2

gcloud run jobs update $JOB_NAME \
    --region=asia-northeast1 \
    --network=tokyo-vpc \
    --subnet=tokyo-subnet \
    --vpc-egress=all-traffic \
    --image=$IMAGE_URL
