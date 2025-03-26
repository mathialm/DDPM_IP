#!/bin/sh

export ENV_MODEL_DIR="/cluster/home/mathialm/poisoning/ML_Poisoning/models/extended_dataset/COCO_TRAFFIC_ext/DDPM-IP/clean/noDef/"
export SLURM_ARRAY_TASK_ID=1
export ENV_GPUS=2
export ENV_DATASET="/cluster/home/mathialm/poisoning/ML_Poisoning/data/datasets32/COCO_TRAFFIC_ext/clean/train.zip"
export ENV_LOG_PATH="./test_mpi"
export ENV_SIZE=32
export ENV_BATCH_SIZE=128
export ENV_MAX_STEPS=1000000
export ENV_SAVE_INTERVAL=20000
export ENV_NODES=1


source activate ADM
echo "Activated environment"
echo "Current CONDA environment: $CONDA_DEFAULT_ENV"

nvidia-smi

mkdir $ENV_MODEL_DIR/$SLURM_ARRAY_TASK_ID

export PYTHONPATH=/cluster/home/mathialm/poisoning/ML_Poisoning/Diffusion_GAN/diffusion_stylegan2
#export NCCL_DEBUG=INFO

cd /cluster/home/mathialm/poisoning/ML_Poisoning/DDPM_IP

echo "Starting application with mpiexec"
#Missing -launcher slurm -hosts $(hostname) parameters from real
mpiexec -launcher slurm -hosts $(hostname) -n $ENV_GPUS python scripts/image_train.py --input_pertub 0.1 \
--data_dir $ENV_DATASET \
--save_path $ENV_MODEL_DIR/$SLURM_ARRAY_TASK_ID \
--log_path $ENV_LOG_PATH/$SLURM_ARRAY_TASK_ID \
--image_size $ENV_SIZE --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size $ENV_BATCH_SIZE --save_interval $ENV_SAVE_INTERVAL \
--resume_checkpoint MAX --lr_anneal_steps $ENV_MAX_STEPS
