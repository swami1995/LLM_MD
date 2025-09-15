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

# python info_main.py --use_chat_api --num_steps 60 --exp_name pro_calibouts | tee outputs/llm_out_pro_calibouts.txt

# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name scratch_trustreg4x_nstep60 | tee outputs/scratch_trustreg4x_nstep60.txt
# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 30 --save_detailed_logs --exp_name bayes_updates_with_ratings_userrep_opt --use_cached_evaluations --cached_evaluations_path run_logs/bayes_updates_with_ratings_cached_fixed_20250828_210510.json | tee outputs/bayes_updates_with_ratings_userrep_opt.txt
# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name scratch_trustreg_nstep60_userforget --use_cached_evaluations --cached_evaluations_path run_logs/scratch_trustreg_nstep60_20250902_231130.json | tee outputs/scratch_trustreg_nstep60_userforget.txt
# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name scratch_trustreg8xuser4x_nstep60_reghor1 --use_cached_evaluations --cached_evaluations_path run_logs/scratch_trustreg4x_nstep60_20250904_012950.json | tee outputs/scratch_trustreg8xuser4x_nstep60_reghor1.txt
# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name scratch_trustreg8xuser4x_nstep60 --use_cached_evaluations --cached_evaluations_path run_logs/scratch_trustreg4x_nstep60_20250904_012950.json | tee outputs/scratch_trustreg8xuser4x_nstep60.txt
# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 100 --save_detailed_logs --exp_name resume_nstep100_trustreg8xuser4x --use_cached_evaluations --cached_evaluations_path run_logs/scratch_trustreg4x_nstep60_20250904_012950.json | tee outputs/resume_nstep100_trustreg8xuser4x.txt 

# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 100 --save_detailed_logs --exp_name resume_nstep100_openaireg | tee outputs/resume_nstep100_openaireg.txt 

# python info_main.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name direct_investing_ur_ad --use_cached_evaluations --cached_evaluations_path run_logs/scratch_trustreg4x_nstep60_20250904_012950.json | tee outputs/direct_investing_ur_ad.txt

# python info_main_all.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name all_main_test_nstep60_selffeedfix --use_cached_evaluations --cached_evaluations_path run_logs/scratch_trustreg4x_nstep60_20250904_012950.json | tee outputs/all_main_test_nstep60_selffeedfix.txt 
# python info_main_all.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name all_main_auditsfixcap_nstep60 --use_cached_evaluations --cached_evaluations_path run_logs/all_main_test_nstep60_selffeedfix_20250908_142833.json | tee outputs/all_main_auditsfixcap_nstep60.txt 

# python info_main_all.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name all_main_auditsfixcap_nstep60 --use_cached_evaluations --cached_evaluations_path run_logs/all_main_audits_nstep60_20250908_180503.json | tee outputs/all_main_auditsfixcap_nstep60.txt 
# python info_main_all.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name all_main_userrep_nstep60 --use_cached_evaluations --cached_evaluations_path run_logs/all_main_test_nstep60_selffeedfix_20250908_142833.json | tee outputs/all_main_userrep_nstep60.txt 

python info_main_all.py --use_chat_api --use_stored_conversations --stored_data_path run_logs/debug_runs/primary_source_test_rounds5_20250708_204246.json --num_steps 60 --save_detailed_logs --exp_name primary_sources_1x2x_usamp07_R33initT20 --use_cached_evaluations --cached_evaluations_path run_logs/resume_nstep100_trustreg8xuser4x_20250904_223242.json | tee outputs/primary_sources_1x2x_usamp07_R33initT20.txt 