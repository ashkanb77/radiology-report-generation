import torch
from tqdm import tqdm
from utils import read_dataset, collate_fn
from dataset import ImageCaptioningDataset
from argparse import ArgumentParser
from model import CaptioningModule
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

parser = ArgumentParser()

parser.add_argument('--processor', type=str, default='Salesforce/blip-image-captioning-base', help='processor')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--model_path', type=str, default='Salesforce/blip-image-captioning-base', help='model')
parser.add_argument('--model_checkpoint', type=str, default=None, help='checkpoint directory')
parser.add_argument('--images_dir', type=str, default=None, help='images directory')
parser.add_argument('--reports_dir', type=str, default=None, help='reports directory')
parser.add_argument('--max_length', type=int, default=200, help='max length of inputs')


args = parser.parse_args()
processor = AutoProcessor.from_pretrained(args.processor_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CaptioningModule(args.model_path)

if args.model_checkpoint is not None:
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))

model.to(device)

rouge = Rouge()


def generate_compute(input_ids, pixel_values):

    generated_ids = model.generate(pixel_values=pixel_values, max_length=150)
    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    ground_truth_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'BLEU-1': 0, 'BLEU-2': 0}

    for i in range(len(generated_captions)):
        reference = [
            ground_truth_captions[i].split()
        ]
        candidate = generated_captions[i].split()

        scores['BLEU-1'] += sentence_bleu(reference, candidate, weights=(1, 0))
        scores['BLEU-2'] += sentence_bleu(reference, candidate, weights=(0, 1))

    scores['BLEU-1'] = scores['BLEU-1'] / (i + 1)
    scores['BLEU-2'] = scores['BLEU-2'] / (i + 1)

    rouge_output = rouge.get_scores(generated_captions, ground_truth_captions)
    for row in rouge_output:
        scores['rouge-1'] += row['rouge-1']['f']
        scores['rouge-2'] += row['rouge-2']['f']
        scores['rouge-l'] += row['rouge-l']['f']

    scores['rouge-1'] = scores['rouge-1'] / len(rouge_output)
    scores['rouge-2'] = scores['rouge-2'] / len(rouge_output)
    scores['rouge-l'] = scores['rouge-l'] / len(rouge_output)

    return generated_captions, ground_truth_captions, scores


_, _, val_image, val_reports = read_dataset(args.images_dir, args.reports_dir)

val_dataset = ImageCaptioningDataset(val_image, val_reports)
val_dataloader = DataLoader(
    val_dataset, collate_fn=lambda data: collate_fn(data, processor, args.max_length),
    batch_size=args.mini_batch_size
)


d = {'prediction': [], 'ground_truth': []}
scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'BLEU-1': 0, 'BLEU-2': 0}
i = 0
with tqdm(val_dataloader, unit="batch") as tepoch:
    for batch in tepoch:
        tepoch.set_description(f"Validation")

        generated_captions, ground_truth_captions, rouge_output = generate_compute(
            input_ids=batch['input_ids'].to(device),
            pixel_values=batch['pixel_values'].to(device)
        )

        scores = {k: scores[k] + v for k, v in rouge_output.items()}
        i += 1

        d['prediction'] = d['prediction'] + generated_captions
        d['ground_truth'] = d['ground_truth'] + ground_truth_captions
        tepoch.set_postfix(rouge1=rouge_output['rouge-1'], rouge2=rouge_output['rouge-2'],
                           rougel=rouge_output['rouge-l'])

scores = {k: v / i for k, v in scores.items()}
print(scores)
df = pd.DataFrame(d)
df.to_csv('validation.csv')
