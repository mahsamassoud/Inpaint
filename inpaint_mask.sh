#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=run_vanilla_diff
#SBATCH --output=logs/%j_out.txt
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
source ~/diffusion/bin/activate
cd /home/mila/m/mahsa.massoud/VanillaDM/improved-diffusion
# module --ignore-cache load cuda/11.1.1
echo "Starting job"
export OPENAI_LOGDIR='/home/mila/m/mahsa.massoud/VanillaDM/improved-diffusion/logs/'
accelerate launch deconv.py --train_data_dir '/home/mila/m/mahsa.massoud/VanillaDM/improved-diffusion/datasets/Masked_RFI_4' --output_dir="ddpm_result/outputs_2" --num_epochs=5 --prediction_type="sample"
echo "Done"