 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/input-perturbation-reduces-exposure-bias-in/image-generation-on-ffhq-128-x-128)](https://paperswithcode.com/sota/image-generation-on-ffhq-128-x-128?p=input-perturbation-reduces-exposure-bias-in)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/input-perturbation-reduces-exposure-bias-in/image-generation-on-celeba-64x64)](https://paperswithcode.com/sota/image-generation-on-celeba-64x64?p=input-perturbation-reduces-exposure-bias-in)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/input-perturbation-reduces-exposure-bias-in/image-generation-on-lsun-tower-64x64)](https://paperswithcode.com/sota/image-generation-on-lsun-tower-64x64?p=input-perturbation-reduces-exposure-bias-in)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/input-perturbation-reduces-exposure-bias-in/image-generation-on-imagenet-32x32)](https://paperswithcode.com/sota/image-generation-on-imagenet-32x32?p=input-perturbation-reduces-exposure-bias-in)


## DDPM-IP
This is the codebase for the ICML 2023 paper [**Input Perturbation Reduces Exposure Bias in Diffusion Models**](https://arxiv.org/abs/2301.11706).
<br>This repository is heavily based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with training modification of input perturbation.

Also, feel free to check out our new paper [**Elucidating the Exposure Bias in Diffusion Models**](https://arxiv.org/abs/2308.15321) which introduces a **simple training-free** solution to exposure bias. Repository: [ADM-ES](https://github.com/forever208/ADM-ES) and [EDM-ES](https://github.com/forever208/EDM-ES)

## Simple to implement Input Perturbation in diffusion models
Our proposed __Input Perturbation__  is an extremely simple __plug-in__ method for general diffusion models.
The implementation of Input Perturbation is just two lines of code.

For instance, based on [guided-diffusion](https://github.com/openai/guided-diffusion),
the only code modifications are in the script [guided_diffusion/gaussian_diffusion.py](https://github.com/forever208/DDPM-IP/blob/DDPM-IP/guided_diffusion/gaussian_diffusion.py), in line 765-766:

```python
new_noise = noise + gamma * th.randn_like(noise)  # gamma=0.1
x_t = self.q_sample(x_start, t, noise=new_noise)
```
NOTE THAT: change the parameter `GPUS_PER_NODE = 4` in the script `dist_util.py` according to your GPU cluster configuration.


## Installation
the installation is the same with [guided-diffusion](https://github.com/openai/guided-diffusion)
```
git clone https://github.com/forever208/DDPM-IP.git
cd DDPM-IP
conda create -n ADM python=3.8
conda activate ADM
pip install -e .
(note that, pytorch 1.10~1.13 is recommended as our experiments in paper were done with pytorch 1.10 and pytorch 2.0 has not been tested by us in this repo)

# install the missing packages
conda install mpi4py
conda install numpy
pip install Pillow
pip install opencv-python
```


## Download ADM-IP models and ADM base models

We have released checkpoints for the main models in the paper.

(The baseline checkpoint of ImageNet-32 and CelebA-64 are missing due to unexpected server file deletion.
If you have trained the ADM base models, welcome to share the checkpoints)

Here are the download links for model checkpoints:

 * CIFAR10 32x32: [ADM-IP.pt](https://drive.google.com/file/d/107CyhiRazNZWkksvoyYRLn71Pcs5FOBo/view?usp=sharing), [ADM-baseline.pt](https://drive.google.com/file/d/1m-a3WqfpMTibzuTC2om046yWQr20GT0I/view?usp=sharing)
 * ImageNet 32x32: [ADM-IP.pt](https://drive.google.com/file/d/1FFUJDk-__9y9DnAG6DKDx5W7LgEIuJyk/view?usp=share_link)
 * LSUN tower 64x64: [ADM-IP.pt](https://drive.google.com/file/d/1QUaY94bSAiTdGu5T_GtdIXMvGT3X1GMy/view?usp=sharing), [ADM-baseline.pt](https://drive.google.com/file/d/1_7gLg44qP4v5bgJNvSo65MtXDZ3xXW39/view?usp=sharing)
 * CelebA 64x64: [ADM-IP.pt](https://drive.google.com/file/d/1Us9zKaIMh8dDlAZVXt3hR2FQYwhxEPYk/view?usp=sharing)
 * FFHQ 128x128: [ADM-IP.pt](https://drive.google.com/file/d/1cadXgH2YYVGGi5os-h6oQAp_8XCTURc7/view?usp=sharing), [ADM-baseline.pt](https://drive.google.com/file/d/1P5Kgyyqp4Zv1RiW116MgBmeX299bGcDw/view?usp=sharing)
 * CIFAR10 32x32: [DDIM-IP](https://drive.google.com/file/d/1TJ8HLO-LsmMS6GDkETIVQthZbUsd92kF/view?usp=sharing) 
 (NOTE THAT we use [DDIM official code](https://github.com/ermongroup/ddim) to do DDIM-IP training and sampling)
 

## Sampling from pre-trained ADM-IP models

To unconditionally sample from these models, you can use the `image_sample.py` scripts.
Sampling from DDPM-IP has no difference with sampling from `openai/guided-diffusion` since DDPM-IP does not change the sampling process.

For example, we sample 50k images using 100 steps from CIFAR10 by: 
```
mpirun python scripts/image_sample.py \
--image_size 32 --timestep_respacing 100 \
--model_path PATH_TO_CHECKPOINT \
--num_channels 128 --num_head_channels 32 --num_res_blocks 3 --attention_resolutions 16,8 \
--resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.3 \
--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True --batch_size 256 --num_samples 50000
```

sample 50k images using 100 steps from LSUN_tower by: 
```
mpirun -n 1 python scripts/image_sample.py \
--image_size 64 --timestep_respacing 100 \
--model_path PATH_TO_CHECKPOINT \
--use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --batch_size 256 --num_samples 50000
```

sample 50k images using 100 steps from FFHQ128 by:
```
mpirun -n 1 python scripts/image_sample.py \
--image_size 128 --timestep_respacing 100 \
--model_path PATH_TO_CHECKPOINT \
--use_fp16 True --num_channels 256 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --batch_size 128 --num_samples 50000
```



## Results

This table summarizes our input perturbation results based on ADM baselines.
Input perturbation shows tremendous training acceleration and much better FID results.

FID computation details: 
- All FIDs are computed using 50K generated samples (unconditional sampling). 
- For CIFAR10 and ImageNet 32x32, we use the whole training data as the reference batch, 
- For LSUN tower 64x64 and CelebA 64x64, we randomly pick up 50k samples from the training set, forming the reference batch   

<p align="left">
  <img src="https://github.com/forever208/DDPM-IP/blob/DDPM-IP/datasets/FID.png" width='100%' height='100%'/>
</p>


This table summarizes our input perturbation results based on DDIM baselines.
<p align="left">
  <img src="https://github.com/forever208/DDPM-IP/blob/DDPM-IP/datasets/DDIM-IP-results.png" width='50%' height='50%'/>
</p>


## Prepare datasets
Please refer to [README.md](https://github.com/forever208/DDPM-IP/tree/DDPM-IP/datasets) for the data preparation.


## Training ADM-IP

Training diffusion models are described in this [repository](https://github.com/openai/improved-diffusion).

Training ADM-IP only requires one more argument `--input perturbation 0.1` (set `--input perturbation 0.0` for the baseline).

NOTE THAT: if you have problems with slurm multi-node training, try the following setting. Let's say training by 16 GPUs on 2 nodes:
```
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8 # 8 gpus for each node
```
instead of specifying `mpiexec -n 16`, you run by `mpirun python script/image_train.py`. (more discussion can be found [here](https://github.com/openai/guided-diffusion/issues/22))

We share the complete arguments of training ADM-IP in the four datasets:

CIFAR10
```
mpiexec -n 2  python scripts/image_train.py --input_pertub 0.15 \
--data_dir PATH_TO_DATASET \
--image_size 32 --use_fp16 True --num_channels 128 --num_head_channels 32 --num_res_blocks 3 \
--attention_resolutions 16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 64
```

ImageNet 32x32 (you can also choose dropout=0.1)
```
mpiexec -n 4  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 32 --use_fp16 True --num_channels 128 --num_head_channels 32 --num_res_blocks 3 \
--attention_resolutions 16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 128
```

LSUN tower 64x64
```
mpiexec -n 16  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 64 --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 16
```

CelebA 64x64
```
mpiexec -n 16  python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 64 --use_fp16 True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 16
```

FFHQ 128x128
```
mpirun -n 16 python scripts/image_train.py --input_pertub 0.1 \
--data_dir PATH_TO_DATASET \
--image_size 128 --use_fp16 True --num_channels 256 --num_head_channels 64 --num_res_blocks 3 \
--attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True \
--learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 8
```

## Citation
If you find our work useful, please feel free to cite by
```
@article{ning2023input,
  title={Input Perturbation Reduces Exposure Bias in Diffusion Models},
  author={Ning, Mang and Sangineto, Enver and Porrello, Angelo and Calderara, Simone and Cucchiara, Rita},
  journal={arXiv preprint arXiv:2301.11706},
  year={2023}
}
```

```
@article{ning2023elucidating,
  title={Elucidating the Exposure Bias in Diffusion Models},
  author={Ning, Mang and Li, Mingxiao and Su, Jianlin and Salah, Albert Ali and Ertugrul, Itir Onal},
  journal={arXiv preprint arXiv:2308.15321},
  year={2023}
}
```
