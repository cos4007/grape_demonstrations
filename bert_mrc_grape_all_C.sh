#!/bin/bash       3090 python env

export CUDA_VISIBLE_DEVICES=0

dataset=bc4chemd
grape=none
shots=25
bert_model_url="/your_path/BiomedNLP_PubMedBERT_large_uncased_abstract"

for epoch in 25; do  
    echo "Run epochs: $epoch"
    echo -e "\n"
    for sample_seed in 42 1337 2021 5555 9999; do
        data_dir_url="/your_path/dataset/${dataset}_${sample_seed}/${grape}_${shots}_${sample_seed}"
        output_dir_url="/your_path/outputs/${dataset}_${sample_seed}/${dataset}_${grape}_shots_${shots}_seed_${sample_seed}_epochs_${epoch}_bs_1_"

        python3 trainer_for_C.py \
        --task_name "bner" \
        --model_type "bert" \
        --overwrite_output_dir \
        --model_name_or_path $bert_model_url \
        --data_dir $data_dir_url \
        --output_dir $output_dir_url \
        --evaluate_during_training \
        --train_max_seq_length 512 \
        --eval_max_seq_length 512 \
        --num_train_epochs $epoch \
        --do_lower_case \
        --loss_type "ce" \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 1 \
        --logging_steps 100000 \
        --save_steps 100000 \
        --seed $sample_seed \
        --learning_rate 3e-5 \
        --do_train \
        --do_eval \
        --do_predict

        echo "Well Done"
        echo -e "\n"
    done
done
