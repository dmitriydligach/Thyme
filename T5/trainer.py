#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
from torch.utils.data import DataLoader
import random, argparse, os
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer)
import datadtr as data

# deterministic determinism
torch.manual_seed(2020)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
random.seed(2020)

def fit(model, train_loader, val_loader, tokenizer):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  optimizer = AdamW(model.parameters())

  for epoch in range(1, args.n_epochs + 1):
    train_loss, num_train_steps = 0, 0
    model.train()

    for batch in train_loader:
      optimizer.zero_grad()

      batch = tuple(t.to(device) for t in batch)
      source_ids, source_mask, target_ids, target_mask = batch

      labels = target_ids
      labels[labels[:, :] == tokenizer.pad_token_id] = -100

      outputs = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=None,
        decoder_attention_mask=target_mask,
        labels=labels)
      loss = outputs[0]

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss = evaluate(model, val_loader, tokenizer)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss))

def evaluate(model, data_loader, tokenizer):
  """Just compute the loss on the validation set"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  total_loss, num_steps = 0, 0
  model.eval()

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    source_ids, source_mask, target_ids, target_mask = batch

    labels = target_ids
    labels[labels[:, :] == tokenizer.pad_token_id] = -100

    with torch.no_grad():
      outputs = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=None,
        decoder_attention_mask=target_mask,
        labels=labels)
      loss = outputs[0]

    total_loss += loss.item()
    num_steps += 1

  average_loss = total_loss / num_steps
  return average_loss

def generate(model, data_loader, tokenizer):
  """Generate outputs for validation set samples"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    source_ids, source_mask, target_ids, target_mask = batch

    # generated tensor: (batch_size, max_output_length)
    predictions = model.generate(
      input_ids=source_ids,
      max_length=args.max_output_length,
      early_stopping=True,
      num_beams=2,
      attention_mask=source_mask)

    inputs = tokenizer.batch_decode(source_ids, skip_special_tokens=True)
    targets = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(
      predictions,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=True)

    # all predictions in this batch
    for i in range(len(predictions)):
      print('[inputs]', inputs[i])
      print('[targets]', targets[i])
      print('[predictions]', predictions[i])
      print()

def main():
  """Fine-tune on summarization data"""

  tokenizer = T5Tokenizer.from_pretrained(args.model_name)

  train_dataset = data.Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='train',
    n_files=args.n_files)
  train_data_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=args.batch_size)

  val_dataset = data.Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='dev',
    n_files=args.n_files)
  val_data_loader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=args.batch_size)

  model = T5ForConditionalGeneration.from_pretrained(args.model_name)
  fit(model, train_data_loader, val_data_loader, tokenizer)
  generate(model, val_data_loader, tokenizer)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xmi_dir=os.path.join(base, 'Thyme/Xmi/'),
    model_name='t5-large',
    max_input_length=100,
    max_output_length=100,
    partition='train',
    n_files='all',
    batch_size=32,
    n_epochs=2)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  main()