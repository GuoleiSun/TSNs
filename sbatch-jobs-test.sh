#!/bin/bash

# module load pigz
# mkdir -p ${TMPDIR}/datasets/
# tar -I pigz -xvf /cluster/work/cvl/guosun/datasets/multi-tasking/taskonomy-sample-model-1-master.tar.gz -C ${TMPDIR}/datasets/
# export DETECTRON2_DATASETS=${TMPDIR}/datasets

save_path=/cluster/work/cvl/guosun/models/multi-tasking/rel1lrmore-iiai-multitask/


# python test1-new2-2-kevis-loader-multi-in.py --batch_size 40 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-7 --best 129 \



# python test1-new2-2-kevis-loader.py --batch_size 40 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20-repro --best 129 \

# python test1-new2-2-kevis-loader-multi-in.py --batch_size 40 \
#  --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-6 --best 119 \

 # python test1-new2-2-kevis-loader-multi-in.py --batch_size 40 \
 # --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-5 --best 119 \

 # python test1-new2-2-kevis-loader-multi-in.py --batch_size 40 \
 # --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-multi-in-student-except-norm-aug-bs16-iiai-512-7 --best 119 \

 # python test1-new2-2-kevis-loader.py --batch_size 40 \
 # --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20-repro --best 119 \

  python test.py --batch_size 50 \
 --ckdir ${save_path}/checkpoints-new2-2-kevis-loader-student-except-norm-aug-bs16-iiai-512-n_ones_20 --best 129 \