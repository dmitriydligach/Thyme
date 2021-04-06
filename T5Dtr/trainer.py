#!/usr/bin/env python3

import torch

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments)

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

  train_dataset = data.Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='train',
    n_files=args.n_files,
    xml_ref_dir=None,
    xml_out_dir=None)

  val_dataset = data.Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='dev',
    n_files=args.n_files,
    xml_ref_dir=None,
    xml_out_dir=None)

  training_args = Seq2SeqTrainingArguments(
    output_dir='./Results',
    num_train_epochs=args.n_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./Logs')

  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset)

  trainer.train()
  trainer.save_model(args.model_dir)
  trainer.evaluate()

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xml_ref_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Dev/'),
    xmi_dir=os.path.join(base, 'Thyme/Xmi/'),
    xml_out_dir='./Xml/',
    data_reader='dataset_dtr',
    model_dir='Model/',
    model_name='t5-large',
    max_input_length=200,
    max_output_length=200,
    n_files='all',
    learning_rate=5e-5,
    batch_size=16,
    n_epochs=2)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  main()
