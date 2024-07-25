#!/bin/bash

RUNS_NSYS=0

JOB=safetensors0.0
BATCH_SIZE=20
USES_PAD=true
USES_DOWNSCALE=true
USES_RANDOM_PATCH=true
USES_RANDOM_ROTATE=true
USES_RANDOM_SHIFT=true
USES_INSTANCE_NORM=true

SEG_SIZE=$((BATCH_SIZE * 60))

python launch_job.slurm.py \
job=$JOB \
auto_submit=false \
sbatch_config.trainer=train.fsdp.py \
train_config.checkpoint.prefix=$JOB \
train_config.checkpoint.state_dict_type=full \
train_config.checkpoint.preempt_chkpt_saving_iterations=null \
train_config.checkpoint.chkpt_saving_iterations=1 \
train_config.dataset.num_workers=2 \
train_config.dataset.prefetch_factor=10 \
train_config.dataset.pin_memory=true \
train_config.dataset.seg_size=$SEG_SIZE \
train_config.loss.grad_accum_steps=10 \
train_config.dataset.batch_size=$BATCH_SIZE \
train_config.dataset.transforms.set.pad=$USES_PAD \
train_config.dataset.transforms.set.downscale=$USES_DOWNSCALE \
train_config.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
train_config.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
train_config.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
train_config.dataset.transforms.set.instance_norm=$USES_INSTANCE_NORM \
train_config.optim.lr=0.0003 \
train_config.optim.fused=false \
train_config.misc.monitors_dynamics=false \
train_config.misc.compiles_model=false \
train_config.misc.max_eval_iter=10 \
train_config.misc.data_dump_on=true \
train_config.lr_scheduler.warmup_iterations=10 \
train_config.lr_scheduler.total_iterations=1000000 \
train_config.logging.prefix=$JOB \
train_config.dist.dtype=bfloat16

base_command="mpirun -n 4 python train.fsdp.py experiments/yaml/$JOB.yaml"
final_command="OMP_NUM_THREADS=1 "

if [ $RUNS_NSYS -eq 1 ]; then
    final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
fi
final_command+="$base_command"

eval $final_command
