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
random.seed(2020)

# new tokens to be added to tokenizer
new_tokens = ['<t>', '</t>', '<e>', '</e>']

def main():
  """Fine-tune on summarization data"""

  # import data provider (e.g. dtr, rel, or events)
  data = importlib.import_module(args.data_reader)

  # need this to save a fine-tuned model
  if os.path.isdir(args.model_dir):
    shutil.rmtree(args.model_dir)
  os.mkdir(args.model_dir)

  # load pretrained T5 tokenizer
  tokenizer = T5Tokenizer.from_pretrained(args.model_name)

  # load a pretrained T5 model
  model = T5ForConditionalGeneration.from_pretrained(args.model_name)

  # add event markers to tokenizer
  tokenizer.add_tokens(new_tokens)
  model.resize_token_embeddings(len(tokenizer))

  train_dataset = data.Data(
    xml_dir=args.xml_train_dir,
    text_dir=args.text_train_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)

  test_dataset = data.Data(
    xml_dir=args.xml_test_dir,
    text_dir=args.text_test_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)

  training_args = Seq2SeqTrainingArguments(
    output_dir='./Results',
    num_train_epochs=args.n_epochs,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.gener_batch_size,
    learning_rate=args.learning_rate,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./Logs',
    disable_tqdm=True,
    predict_with_generate=True,
    load_best_model_at_end=False)

  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset)

  trainer.train()
  trainer.save_model(args.model_dir)
  print('done training...')

  # results = trainer.predict(
  #   test_dataset=test_dataset,
  #   max_length=args.max_output_length,
  #   num_beams=args.num_beams)
  #
  # predictions = tokenizer.batch_decode(
  #   results.predictions,
  #   skip_special_tokens=True,
  #   clean_up_tokenization_spaces=True)
  #
  # for prediction in predictions:
  #   print(prediction)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xml_train_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Train/'),
    text_train_dir=os.path.join(base, 'Thyme/Text/train/'),
    xml_test_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Dev/'),
    text_test_dir=os.path.join(base, 'Thyme/Text/dev/'),
    xml_out_dir='./Xml/',
    xml_regex='.*[.]Temporal.*[.]xml',
    data_reader='dataset_rel',
    model_dir='Model/',
    model_name='t5-base',
    chunk_size=200,
    max_input_length=512,
    max_output_length=512,
    n_files='all',
    learning_rate=1e-4,
    train_batch_size=12,
    gener_batch_size=128,
    num_beams=3,
    weight_decay=0.0001,
    print_predictions=False,
    print_metadata=False,
    print_errors=False,
    do_train=True,
    early_stopping=True,
    n_epochs=2)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  main()
