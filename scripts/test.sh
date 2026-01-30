#!/bin/bash
export PYTHONUNBUFFERED=1
# export HF_HOME=/path/to/config/bert-base-uncased/
# export TORCH_HOME=/path/to/config/blip2_pretrained_vitL/

python main.py \
  --gpu_num "0" \
  --method "second" \
  --dataset "NYU" \
  --auto_bins True \
  --data_root_path "./datasets/nyu_data/" \
  --test_file "./datasets/nyu_data/data/nyu2_test.csv" \
  --class_name "all" \
  --test_log_save \
  --log_result_dir "./log_results" \
  --model_load_path "./checkpoints/second_NYU_bins_True/train/1/model_epoch_50.pth"

