#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments)

import random, argparse, os, shutil, importlib

# deterministic determinism
torch.manual_seed(2020)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
random.seed(2020)

def main():
  """Fine-tune on summarization data"""

  # need this to save a fine-tuned model
  if os.path.isdir(args.model_dir):
    shutil.rmtree(args.model_dir)
  os.mkdir(args.model_dir)

  # import data provider (e.g. dtr, rel, or events)
  data = importlib.import_module(args.data_reader)

  # load pretrained T5 tokenizer
  tokenizer = T5Tokenizer.from_pretrained(args.model_name)

  # load a pretrained T5 model
  model = T5ForConditionalGeneration.from_pretrained(args.model_name)

  train_dataset = data.Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='train',
    n_files=args.n_files)

  val_dataset = data.Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='dev',
    n_files=args.n_files)

  training_args = TrainingArguments(
    output_dir='./Results',
    num_train_epochs=args.n_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./Logs')

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset)

  trainer.train()
  trainer.save_model(args.model_dir)
  trainer.evaluate()

  # load the saved model
  model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xmi_dir=os.path.join(base, 'Thyme/Xmi/'),
    data_reader='dtr',
    model_dir='Model/',
    model_name='t5-large',
    max_input_length=100,
    max_output_length=100,
    n_files=10,
    learning_rate=1e-4,
    batch_size=32,
    n_epochs=2)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  main()
