python3 covidence_data_aug.py
for SEED in {0..4}
do
  python3 utils/run_multi_class_classifier_mi.py \
    --do_train \
    --do_eval \
    --do_predict \
    --extra-files stage\
    --task_name=covidence \
    --data_dir=./data/text_classification/covidence_ensemble$SEED \
    --model_name_or_path=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract\
    --output_dir=./output/covidence_ensemble$SEED \
    --learning_rate=1e-5 \
    --num_train_epochs=4 \
    --logging_steps=0 \
    --save_steps=0 \
    --model_type=bert \
    --max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --seed=0 \
    --do_lower_case \
    --overwrite_output_dir \
    --fp16 \
    --weight_decay 15
done