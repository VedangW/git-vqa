import os
import json
import base64
import argparse
from tqdm import tqdm

from load_aokvqa import load_aokvqa, get_coco_path

from generativeimage2text.common import read_to_buffer
from generativeimage2text.tsv_io import tsv_writer


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa_dir', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--coco_dir', type=str, required=True)

parser.add_argument('--tsv_dir', type=str, required=True)
parser.add_argument('--images_tsv', type=str, required=True)
parser.add_argument('--questions_tsv', type=str, required=True)
args = parser.parse_args()

print(f"TSV directory: {args.tsv_dir}.")

image_fname = args.images_tsv + '_' + args.split + '.tsv'
question_fname = args.questions_tsv + '_' + args.split + '.tsv'

images_tsv = args.tsv_dir + '/' + image_fname
questions_tsv = args.tsv_dir + '/' + question_fname

data = load_aokvqa(args.aokvqa_dir, args.split)

def gen_image_rows():
    for sample in tqdm(data):
        payload = base64.b64encode(read_to_buffer(get_coco_path(args.split, sample['image_id'], args.coco_dir)))
        yield sample['image_id'], payload


def gen_question_rows():
    for sample in tqdm(data):
        qs = [{'question_id': sample['question_id'], 'question': sample['question']}]
        yield sample['image_id'], json.dumps(qs)

tsv_writer(gen_image_rows(), images_tsv)
tsv_writer(gen_question_rows(), questions_tsv)

print("Done!")
