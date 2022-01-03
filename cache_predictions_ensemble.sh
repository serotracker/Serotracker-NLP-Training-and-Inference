eval "$(grep ^USERNAME= .env)"
eval "$(grep ^PASSWORD= .env)"

# download csvs from covidence
python3 download_csvs_from_covidence.py --username $USERNAME --password $PASSWORD --mode inference

preprocess text and run inference for each of 5 members in ensemble
for SEED in {0..4}
do
    rm ./data/text_classification/covidence_ensemble$SEED/cached_all_*
    python3 prepare_text_for_prediction.py --files ./csvs_for_inference/title_and_abstract_screening.csv --output_dir covidence_ensemble$SEED
    python3 utils/run_multi_class_classifier_mi.py \
    --do_predict \
    --extra-files all stage\
    --task_name=covidence \
    --data_dir=./data/text_classification/covidence_ensemble$SEED  \
    --model_name_or_path=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract\
    --output_dir=./output/covidence_ensemble$SEED\
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

#make predictions for PIO, combine them with inclusion recommendations and post to the server
python3 make_pio_predictions.py --files ./csvs_for_inference/title_and_abstract_screening.csv
python3 combine_prediction_files.py --files ./csvs_for_inference/title_and_abstract_screening.csv
python3 post_predictions.py

sleep 5

python3 run_opot.py --username $USERNAME --password $PASSWORD
