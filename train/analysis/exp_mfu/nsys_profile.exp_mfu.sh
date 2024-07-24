#!/bin/bash

RUNS_NSYS=0

JOB=exp0.1
INPUT_H=512
INPUT_W=512
BATCH_SIZE=80
USES_PAD=false
USES_DOWNSCALE=false
USES_RANDOM_PATCH=false
USES_RANDOM_ROTATE=false
USES_RANDOM_SHIFT=false
USES_INSTANCE_NORM=true

SEG_SIZE=$((BATCH_SIZE * 60))
TOTAL_SIZE=$((BATCH_SIZE * 1000))

python launch_job.slurm.exp_mfu.py \
job=$JOB \
auto_submit=false \
sbatch_config.trainer=train.fsdp.dummy_dataset.py \
exp_mfu.checkpoint.prefix=$JOB \
exp_mfu.checkpoint.state_dict_type=full \
exp_mfu.checkpoint.preempt_chkpt_saving_iterations=null \
exp_mfu.checkpoint.chkpt_saving_iterations=null \
exp_mfu.dataset.num_workers=4 \
exp_mfu.dataset.prefetch_factor=20 \
exp_mfu.dataset.pin_memory=true \
exp_mfu.dataset.seg_size=$SEG_SIZE \
exp_mfu.loss.grad_accum_steps=10 \
exp_mfu.dataset.batch_size=$BATCH_SIZE \
exp_mfu.dataset.input.H=$INPUT_H \
exp_mfu.dataset.input.W=$INPUT_W \
exp_mfu.dataset.input.total_size=$TOTAL_SIZE \
exp_mfu.dataset.transforms.set.pad=$USES_PAD \
exp_mfu.dataset.transforms.set.downscale=$USES_DOWNSCALE \
exp_mfu.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
exp_mfu.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
exp_mfu.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
exp_mfu.dataset.transforms.set.instance_norm=$USES_INSTANCE_NORM \
exp_mfu.optim.lr=0.0003 \
exp_mfu.optim.fused=false \
exp_mfu.misc.monitors_dynamics=false \
exp_mfu.misc.compiles_model=false \
exp_mfu.misc.max_eval_iter=10 \
exp_mfu.lr_scheduler.warmup_iterations=10 \
exp_mfu.lr_scheduler.total_iterations=1000000 \
exp_mfu.logging.prefix=$JOB \
exp_mfu.dist.dtype=bfloat16 \
exp_mfu.model.backbone.hf_config.image_size=$INPUT_H

base_command="mpirun -n 4 python train.fsdp.dummy_dataset.py experiments/yaml/$JOB.yaml"
final_command="OMP_NUM_THREADS=1 "

if [ $RUNS_NSYS -eq 1 ]; then
    final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
fi
final_command+="$base_command"

eval $final_command
