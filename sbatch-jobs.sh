#!/bin/bash

# module load pigz
# mkdir -p ${TMPDIR}/datasets/
# tar -I pigz -xvf /cluster/work/cvl/guosun/datasets/multi-tasking/taskonomy-sample-model-1-master.tar.gz -C ${TMPDIR}/datasets/
# export DETECTRON2_DATASETS=${TMPDIR}/datasets

save_path=/cluster/work/cvl/guosun/models/multi-tasking/rel1lrmore-iiai-multitask/


# python train-new2-2-kevis-loader-multi-in.py  --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-4 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-4

## from here, change to leakyrelu

# python train-new2-2-kevis-loader-multi-in.py  --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-5 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-5

# python train-new2-2-kevis-loader-multi-in.py  --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-6 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-6

# python train-new2-2-kevis-loader-multi-in.py  --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-7 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-7

 # python train-new2-2-kevis-loader.py --batch_size 16 \
 #  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20-repro \
 #   --logdir logs2/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20-repro

# python train-new2-2-kevis-loader-multi-bn.py  --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-bn-student-except-norm-aug-bs16-iiai-512-3 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-multi-bn-student-except-norm-aug-bs16-iiai-512-3


python train-new2-2-kevis-loader-nyud.py --batch_size 16 \
 --ckdir checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20-nyud \
  --logdir logs2/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20-nyud




