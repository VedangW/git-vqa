import json
import torch
import argparse
from tqdm import tqdm
from load_aokvqa import load_aokvqa
from sentence_transformers import SentenceTransformer, util

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--aokvqa_dir', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

data = load_aokvqa(args.aokvqa_dir, args.split)
choices = {k['question_id']: k['choices'] for k in data}

st = SentenceTransformer('all-MiniLM-L6-v2')

with open(args.input, 'r') as f:
    preds = f.readlines()

out = {}
for p in tqdm(preds):
    p = json.loads(p)

    choice_embeddings = [st.encode(choice) for choice in choices[p['question_id']]]
    out_embedding = st.encode(p['answer'])

    cos_scores = torch.tensor([util.cos_sim(x, out_embedding) for x in choice_embeddings])
    chosen_choice = choices[p['question_id']][torch.argmax(cos_scores).item()]

    out[p['question_id']] = {
        'direct_answer': p['answer'],
        'multiple_choice': chosen_choice,
    }


with open(args.output, 'w') as f:
    json.dump(out, f)
