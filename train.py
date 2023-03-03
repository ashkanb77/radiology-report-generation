import torch
from tqdm import tqdm
from utils import read_dataset, collate_fn
from dataset import ImageCaptioningDataset
from argparse import ArgumentParser
from model import CaptioningModule
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import time
import gc
import os


parser = ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--mini_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--max_length', type=int, default=200, help='maximum length of captions')
parser.add_argument('--processor', type=str, default="Salesforce/blip-image-captioning-base", help='processor')
parser.add_argument('--model_path', type=str, default="Salesforce/blip-image-captioning-base", help='model')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='checkpoint directory')
parser.add_argument('--reports_dir', type=str, default='ecgen-radiology', help='reports directory')
parser.add_argument('--images_dir', type=str, default='images', help='images directory')
parser.add_argument('--resume_training_dir', type=str, default=None, help='dataset directory')
parser.add_argument('--dtype', type=str, default='float32', help='floating point')

args = parser.parse_args()

if args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
else:
    dtype = torch.float32


experiment = str(int(time.time()))
dir = os.path.join(args.checkpoint_dir, experiment)
os.makedirs(dir, exist_ok=True)
writer = SummaryWriter(os.path.join(dir, 'runs'))

train_images, train_reports, test_images, test_reports = read_dataset(
    args.images_dir, args.reports_dir
)

processor = AutoProcessor.from_pretrained(args.processor)

train_dataset = ImageCaptioningDataset(train_images, train_reports)
val_dataset = ImageCaptioningDataset(test_images, test_reports)

train_dataloader = DataLoader(
    train_dataset, collate_fn=lambda data: collate_fn(data, processor, args.max_length),
    batch_size=args.mini_batch_size
)
val_dataloader = DataLoader(
    val_dataset, collate_fn=lambda data: collate_fn(data, processor, args.max_length),
    batch_size=args.mini_batch_size
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CaptioningModule(args.model_path)
model.to(device)

# training config: optimizer, scheduler and criterion
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0,
    num_training_steps=args.epochs
)

acc_steps = args.batch_size // args.mini_batch_size
scaler = torch.cuda.amp.GradScaler()


def train_epoch():
    losses = []
    model.train()

    mini_loss = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for i, batch in enumerate(tepoch):
            tepoch.set_description(f"Train Epoch {epoch + 1}/{args.epochs}")

            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            with autocast(dtype=dtype):
                output = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )

            loss = output['loss']
            mini_loss += loss.item()
            scaler.scale(loss).backward()

            if i % acc_steps == (acc_steps - 1):
                losses.append(mini_loss / acc_steps)

                optimizer.step()
                optimizer.zero_grad()

            tepoch.set_postfix(loss=loss.item())

    scheduler.step()
    loss = np.mean(losses)

    writer.add_scalar('Loss/train', loss, epoch)
    return loss


def eval_model():
    losses = []
    model.eval()

    with torch.no_grad():
        with tqdm(val_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Validation Epoch {epoch + 1}/{args.epochs}")

                input_ids = batch['input_ids'].to(device)
                pixel_values = batch['pixel_values'].to(device)

                output = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                )

                loss = output['loss']

                losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        loss = np.mean(losses)
        writer.add_scalar('Loss/validation', loss, epoch)
        return loss


losses = []
val_losses = []

best_loss = 100

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

if args.resume_training_dir is not None:
    model.load_state_dict(torch.load(os.path.join(args.resume_training_dir, 'model.pth')))
    optimizer.load_state_dict(torch.load(os.path.join(args.resume_training_dir, 'optimizer.pth')))
    scheduler.load_state_dict(torch.load(os.path.join(args.resume_training_dir, 'scheduler.pth')))
    scaler.load_state_dict(torch.load(os.path.join(args.resume_training_dir, 'scaler.pth')))

for epoch in range(args.epochs):

    # train one epoch
    train_loss = train_epoch()

    print(f'Train loss {train_loss:0.4f}')

    # evaluate
    val_loss = eval_model()

    print(f'Validation loss {val_loss:0.4f}')
    print()

    # save history
    losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss:  # save model if its accuracy is bigger than best model accuracy
        torch.save(model.state_dict(), os.path.join(dir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(dir, 'optimizer.pth'))
        torch.save(scheduler.state_dict(), os.path.join(dir, 'scheduler.pth'))
        torch.save(scaler.state_dict(), os.path.join(dir, 'scaler.pth'))
        best_loss = val_loss

torch.save(optimizer.state_dict(), os.path.join(dir, 'optimizer.pth'))
torch.save(scheduler.state_dict(), os.path.join(dir, 'scheduler.pth'))
torch.save(scaler.state_dict(), os.path.join(dir, 'scaler.pth'))

print(f"Best Loss is {best_loss:0.4f}")
