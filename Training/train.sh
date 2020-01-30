#! /bin/bash

#The private IPs of the AWS instances to be used for training the model.
#List the local instance (the one on which this script is running) first.
#The format is ("instance1" "instance2", ...)
REMOTE_HOSTS=("172.31.61.99" "172.31.59.203")

#The number of GPUs to use on each instance.
#Use the same order as for HOSTS.
GPUS=(1 1)

#The location of the data for training.
export DATA_FOLDER="/home/ubuntu/Insight/s3mnt/data2/preprocessed/"

#The location of the model to train
export MODEL="/home/ubuntu/Insight/Vulnerable_Buildings/Models/inception_model2.h5"

#The location to save the trained model
export TRAINED_MODEL="/home/ubuntu/Insight/Vulnerable_Buildings/Models/inception_model2_trained.h5"

#The number of epoches to train
export EPOCHS=1

#The batch size
export BATCH_SIZE=32

TOTAL_GPUS=0
for GPU in ${GPUS[@]}; do
  let TOTAL_GPUS+=$GPU
done

HOST_LIST="localhost:${GPUS[0]}"
for i in ${!REMOTE_HOSTS[@]}; do
    if [ $i -gt 0 ]
    then
        HOST_LIST="$HOST_LIST,${REMOTE_HOSTS[$i]}:${GPUS[$i]}"
    fi
done

horovodrun --verbose -np $TOTAL_GPUS -H $HOST_LIST python train_inception_model.py