#!/bin/bash

# module load pigz
# mkdir -p ${TMPDIR}/datasets/
# tar -I pigz -xvf /cluster/work/cvl/guosun/datasets/multi-tasking/taskonomy-sample-model-1-master.tar.gz -C ${TMPDIR}/datasets/
# export DETECTRON2_DATASETS=${TMPDIR}/datasets

save_path=/cluster/work/cvl/guosun/models/multi-tasking/rel1lrmore-iiai-multitask/


# python train-new2-2-no-adain-kevis-loader-single-task.py --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-seg-3 \
#   --logdir logs2/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-seg-3

# python train-new2-2-no-adain-kevis-loader-single-task.py --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-parts-3 \
#   --logdir logs2/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-parts-3


# python train-new2-2-no-adain-kevis-loader-single-task.py --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-edges-3 \
#   --logdir logs2/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-edges-3


# python train-new2-2-no-adain-kevis-loader-single-task.py --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-normals-3 \
#   --logdir logs2/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-normals-3


# python train-new2-2-no-adain-kevis-loader-single-task.py --batch_size 16 \
#  --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-saliency-3 \
#   --logdir logs2/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-saliency-3


# ## NYUD dataset
# python train-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 8 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-seg-3 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-seg-3

# python train-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 8 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-edges-3 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-edges-3

# python train-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 8 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-normals-3 \
#   --logdir logs2/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-normals-3

python train-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 8 \
 --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-depth-3 \
  --logdir logs2/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-depth-3