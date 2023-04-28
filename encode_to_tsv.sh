# Training set
python encode_to_tsv.py \
    --aokvqa_dir ${AOKVQA_DIR} \
    --split train \
    --coco_dir ${COCO_DIR} \
    --tsv_dir ${DATA_DIR}/aokvqa_base/tsvs \
    --images_tsv images \
    --questions_tsv questions

# Validation set
python encode_to_tsv.py \
    --aokvqa_dir ${AOKVQA_DIR} \
    --split val \
    --coco_dir ${COCO_DIR} \
    --tsv_dir ${DATA_DIR}/aokvqa_base/tsvs \
    --images_tsv images \
    --questions_tsv questions

# Test set
python encode_to_tsv.py \
    --aokvqa_dir ${AOKVQA_DIR} \
    --split test \
    --coco_dir ${COCO_DIR} \
    --tsv_dir ${DATA_DIR}/aokvqa_base/tsvs \
    --images_tsv images \
    --questions_tsv questions