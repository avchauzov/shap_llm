#!/bin/bash

python finetune_LLaVA/llava/eval/run_llava.py \
    --model-path finetune_LLaVA/llava/checkpoints/llama-2-7b-chat-task-qlora/best_llava_eval_model \
    --model-base llava-v1.5-7b \
    --image-file _data/shap_plots_scatter/linear/images/0.png \
    --query "What is shown in the image?"