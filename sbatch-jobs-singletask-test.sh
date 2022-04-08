#!/bin/bash

# module load pigz
# mkdir -p ${TMPDIR}/datasets/
# tar -I pigz -xvf /cluster/work/cvl/guosun/datasets/multi-tasking/taskonomy-sample-model-1-master.tar.gz -C ${TMPDIR}/datasets/
# export DETECTRON2_DATASETS=${TMPDIR}/datasets

save_path=/cluster/work/cvl/guosun/models/multi-tasking/rel1lrmore-iiai-multitask/


python test-single-task.py --batch_size 40 \
 --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-seg-3 --best 129 \

 python test-single-task.py --batch_size 40 \
 --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-parts-3 --best 129 --task parts \


  python test-single-task.py --batch_size 40 \
 --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-normals-3 --best 129 \

 # python test-single-task.py --batch_size 40 \
 # --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-edges-3 --best 129 \

   python test-single-task.py --batch_size 40 \
 --ckdir ${save_path}/checkpoints-new2-2-no-adain-kevis-loader-student-except-norm-aug-bs16-iiai-256-n_ones_20-in-saliency-3 --best 1 --task saliency \


## nyud dataset

# python test1-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 40 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-seg-3 --best 229 \

python test1-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 40 \
 --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs8-iiai-512-n_ones_30-nyud-in-normals-3 --best 229 \

 # python test1-new2-2-no-adain-kevis-loader-single-task-nyud.py --batch_size 40 \
 # --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_30-nyud-in-depth-3 --best 299 \
