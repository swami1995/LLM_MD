#!/bin/bash
#SBATCH --job-name=eval_llmmd
#SBATCH --partition=general
#SBATCH --constraint=L40S
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-24:00:00
#SBATCH --output=outputs/job_output_%j.txt
#SBATCH --error=outputs/job_error_%j.txt


echo Activating conda env...

source ~/.bashrc

conda activate llmmd

python info_main.py --use_chat_api --num_steps 60 --exp_name pro_calibouts | tee outputs/llm_out_pro_calibouts.txt
