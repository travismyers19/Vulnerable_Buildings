#! /bin/bash

#The private IPs of the AWS instances to be used for training the model.
#List the local instance (the one on which this script is running) first.
#The format is ("instance1" "instance2" ...)
#HOSTS=("172.31.61.99" "172.31.59.203")
#HOSTS=("172.31.61.99")
HOSTS=("172.31.59.203")

#The number of GPUs to use on each instance.
#Use the same order as for HOSTS.
GPUS=(1)

#The location of the data for training.
export TRAINING_DIRECTORY="/home/ubuntu/Insight/s3mnt/Training_Soft/"

#The location of the data for testing/validation.
export TEST_DIRECTORY="/home/ubuntu/Insight/s3mnt/Test_Soft_Binary/"

#The location of the model to train
export MODEL_FILENAME="/home/ubuntu/Insight/s3mnt/Models/BinaryFocalLoss/soft_model.h5"

#The location to save the trained model
export TRAINED_MODEL_FILENAME="/home/ubuntu/Insight/s3mnt/Models/BinaryFocalLoss/soft_model_trained_1_5_1GPU.h5"

#The location to save the training metrics (a numpy file)
export METRICS_FILENAME="/home/ubuntu/Insight/s3mnt/Models/BinaryFocalLoss/soft_metrics_1_5_1GPU.npy"

#How much to weigh soft story buildings vs. non-soft (1.0 = full weight, 0.0 = no weight).  "None" if no weight.
export SOFT_WEIGHT=0.2

#How much to weigh good images vs bad images.  "None" if no weight.
export GOOD_WEIGHT=

#Whether or not the model is a binary classifier (as opposed to a ternary classifier)
export BINARY=1

#The number of epochs to train
export EPOCHS=5

#The batch size
export BATCH_SIZE=32

TOTAL_GPUS=0
for GPU in ${GPUS[@]}; do
  let TOTAL_GPUS+=$GPU
done

HOST_LIST="localhost:${GPUS[0]}"
for i in ${!HOSTS[@]}; do
    if [ $i -gt 0 ]
    then
        HOST_LIST="$HOST_LIST,${HOSTS[$i]}:${GPUS[$i]}"
    fi
done

horovodrun --verbose -np $TOTAL_GPUS -H $HOST_LIST python train_inception_model.py