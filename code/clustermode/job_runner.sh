#!/usr/bin/env bash
export AZURE_STORAGE_ACCOUNT=<storage account name>
export AZURE_STORAGE_ACCESS_KEY=<storage account access key>
spark-submit  --master yarn --executor-memory 3G --driver-memory 6G --num-executors 85 --executor-cores 1 randomProjection.py
zip /tmp/srp.pkl.zip /tmp/srp.pkl*
azure storage blob upload --container modelling-outputs --file /tmp/srp.pkl.zip