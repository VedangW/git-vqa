USE_GPU=1
WORLD_SIZE=4
MODEL='GIT_LARGE_VQAv2' # Options: ('GIT_BASE_VQAv2', 'GIT_LARGE_VQAv2')

if [ $USE_GPU -eq 1 ]; then
    AZFUSE_TSV_USE_FUSE=1 mpirun -n ${WORLD_SIZE} python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
        'image_tsv': '${DATA_DIR}/aokvqa_base/tsvs/images_val.tsv', \
        'model_name': ${MODEL}, \
        'question_tsv': '${DATA_DIR}/aokvqa_base/tsvs/questions_val.tsv', \
        'out_tsv': '${DATA_DIR}/aokvqa_base/tsvs/val_preds_${MODEL}.tsv', \
    }"
else
    AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
        'image_tsv': '${DATA_DIR}/aokvqa_base/tsvs/images_val.tsv', \
        'model_name': ${MODEL}, \
        'question_tsv': '${DATA_DIR}/aokvqa_base/tsvs/questions_val.tsv', \
        'out_tsv': '${DATA_DIR}/aokvqa_base/tsvs/val_preds_${MODEL}.tsv', \
    }"
fi

