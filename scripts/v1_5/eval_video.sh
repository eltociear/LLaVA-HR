#!/bin/bash


MODEL_PATH=$1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/seed.sh $1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/activitynet.sh $1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/msrvtt.sh $1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/msvd.sh $1
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_full/sqa.sh $1
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_full/textvqa.sh  $1
bash scripts/v1_5/eval_full/gqa.sh $1
CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/eval_full/mme.sh $1
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_full/pope.sh $1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval_full/vqav2.sh $1
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval_full/vizwiz.sh $1