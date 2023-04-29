MODEL='GIT_LARGE_VQAv2' # Options: ('GIT_BASE_VQAv2', 'GIT_LARGE_VQAv2')

# Transform predictions to json
python transform_output.py \
    --input ${DATA_DIR}/aokvqa_base/tsvs/val_preds_${MODEL}.tsv \
    --aokvqa_dir ${AOKVQA_DIR} \
    --split val \
    --output ${PREDS_DIR}${MODEL}_val.json

# Evaluate predictions
python eval_predictions.py \
    --aokvqa-dir ${AOKVQA_DIR} \
    --split val \
    --preds ${PREDS_DIR}${MODEL}_val.json