import torch
from argparse import ArgumentParser
from model import CaptioningModule
from transformers import AutoProcessor
from PIL import Image
import gc
import os


parser = ArgumentParser()

parser.add_argument('--max_length', type=int, default=200, help='Maximum length of captions')
parser.add_argument('--processor', type=str, default="Salesforce/blip-image-captioning-base", help='processor')
parser.add_argument('--model_path', type=str, default="Salesforce/blip-image-captioning-base", help='model')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='checkpoint directory')
parser.add_argument('--image_path', type=str, default='images', help='images directory')

args = parser.parse_args()


processor = AutoProcessor.from_pretrained(args.processor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CaptioningModule(args.model_path)
model.to(device)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

model.load_state_dict(torch.load(os.path.join(args.resume_training_dir, 'model.pth')))

image = Image.open(args.image_path)

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=150)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
