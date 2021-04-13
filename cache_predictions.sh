python3 abstract_screen_demo/prepare_text_for_prediction.py --files ./csvs/excluded.csv ./csvs/included.csv ./csvs/irrelevant.csv ./csvs/full_text_review.csv ./csvs/title_and_abstract_screening.csv
python abstract_screen_demo/utils/run_multi_class_classifier_mi.py \
  --do_eval \
  --do_predict \
  --extra-files combined all\
  --task_name=covidence \
  --data_dir=./abstract_screen_demo/data/text_classification/covidence_aug \
  --model_name_or_path=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract\
  --output_dir=./output/covidence_aug_mi2 \
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
python3 abstract_screen_demo/make_pio_predictions.py --files ./csvs/excluded.csv ./csvs/included.csv ./csvs/irrelevant.csv ./csvs/full_text_review.csv ./csvs/title_and_abstract_screening.csv
python3 abstract_screen_demo/combine_prediction_files.py --files ./csvs/excluded.csv ./csvs/included.csv ./csvs/irrelevant.csv ./csvs/full_text_review.csv ./csvs/title_and_abstract_screening.csv