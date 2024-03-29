#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import random, argparse, os, shutil, importlib, torch

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup)

from torch.utils.data import DataLoader

# deterministic determinism
# todo: anything else needed here?
torch.manual_seed(2020)
random.seed(2020)

def fit(model, train_loader, val_loader, tokenizer):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  # define parameter groups
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}]

  # implements gradient bias correction as well as weight decay
  # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
  scheduler = get_linear_schedule_with_warmup(optimizer, 100, 1500)

  optimal_epochs = 0
  best_loss = float('inf')

  for epoch in range(1, args.n_epochs + 1):
    train_loss, num_train_steps = 0, 0
    model.train()

    for batch in train_loader:
      optimizer.zero_grad()

      # metadata not needed here
      batch.pop('metadata')

      # tensors in batch to gpu
      for key in batch.keys():
        batch[key] = batch[key].to(device)

      # ignore padding
      batch['labels'][batch['labels'][:, :] == tokenizer.pad_token_id] = -100

      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss = evaluate(model, val_loader, tokenizer)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss))

    if val_loss < best_loss:
      print('loss improved, saving model...')
      model.save_pretrained(args.model_dir)
      best_loss = val_loss
      optimal_epochs = epoch

  return best_loss, optimal_epochs

def evaluate(model, data_loader, tokenizer):
  """Just compute the loss on the validation set"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  total_loss, num_steps = 0, 0
  model.eval()

  for batch in data_loader:

    # metadata not needed here
    batch.pop('metadata')

    # tensors in batch to gpu
    for key in batch.keys():
      batch[key] = batch[key].to(device)

    # ignore padding
    batch['labels'][batch['labels'][:, :] == tokenizer.pad_token_id] = -100

    with torch.no_grad():
      outputs = model(**batch)
      loss = outputs.loss

    total_loss += loss.item()
    num_steps += 1

  average_loss = total_loss / num_steps
  return average_loss

def generate(model, data_loader, tokenizer):
  """Generate outputs for validation set samples"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  # key: (file, start, end), value: prediction
  prediction_lookup = {}

  for batch in data_loader:

    # metadata for this batch
    metadata = batch.pop('metadata')

    for key in batch.keys():
      batch[key] = batch[key].to(device)

    # generated tensor: (batch_size, max_output_length)
    predictions = model.generate(
      input_ids=batch['input_ids'],
      max_length=args.max_output_length,
      early_stopping=True,
      num_beams=args.num_beams,
      attention_mask=batch['attention_mask'],
      decoder_attention_mask=batch['decoder_attention_mask']) # todo: is this necessary?

    inputs = tokenizer.batch_decode(
      batch['input_ids'],
      skip_special_tokens=True)
    targets = tokenizer.batch_decode(
      batch['labels'],
      skip_special_tokens=True)
    predictions = tokenizer.batch_decode(
      predictions,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=True)

    # metadata example for a sentence (i.e. multiple events):
    # ID085_clinic_251|14784|14791||ID085_clinic_251|14809|14819

    # iterate over samples in this batch
    for i in range(len(predictions)):
      if args.print_predictions:
        print('[input]', inputs[i])
        print('[targets]', targets[i])
        print('[predict]', predictions[i])
        print('[metdata]', metadata[i], '\n')

      event_dtr_list = predictions[i].split(', ')
      event_metadata_list = metadata[i].split('||')

      if len(event_dtr_list) != len(event_metadata_list):
        min_length = min(len(event_dtr_list), len(event_metadata_list))
        event_dtr_list = event_dtr_list[:min_length]
        event_metadata_list = event_metadata_list[:min_length]

      for event_dtr, event_metadata in zip(event_dtr_list, event_metadata_list):
        elements = event_dtr.split('|')
        if len(elements) == 2:
          prediction_lookup[event_metadata] = elements[1]

  return prediction_lookup

def perform_fine_tuning():
  """Fine-tune and save model"""

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
  tokenizer.add_tokens(['<e>', '</e>'])
  model.resize_token_embeddings(len(tokenizer))

  train_dataset = data.Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='train',
    n_files=args.n_files,
    xml_ref_dir=None,
    xml_out_dir=None)
  train_data_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=args.train_batch_size)

  val_dataset = data.Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='dev',
    n_files=args.n_files,
    xml_ref_dir=None,
    xml_out_dir=None)
  val_data_loader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=args.train_batch_size)

  # fine-tune model on thyme data and save it
  best_loss, optimal_epochs = fit(
    model,
    train_data_loader,
    val_data_loader,
    tokenizer)
  print('best loss %.3f after %d epochs\n' % (best_loss, optimal_epochs))

def perform_generation():
  """Load fine-tuned model and generate"""

  # import data provider (e.g. dtr, rel, or events)
  data = importlib.import_module(args.data_reader)

  # load pretrained T5 tokenizer
  tokenizer = T5Tokenizer.from_pretrained(args.model_name)

  # load the saved model
  model = T5ForConditionalGeneration.from_pretrained(args.model_dir)

  # add event markers to tokenizer
  tokenizer.add_tokens(['<e>', '</e>'])
  model.resize_token_embeddings(len(tokenizer))

  val_dataset = data.Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition='dev',
    n_files=args.n_files,
    xml_ref_dir=args.xml_ref_dir,
    xml_out_dir=args.xml_out_dir)
  val_data_loader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=args.gener_batch_size)

  # generate output from the saved model
  prediction_lookup = generate(model, val_data_loader, tokenizer)

  # write anafora xml for evaluation
  val_dataset.write_xml(prediction_lookup)

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
    learning_rate=1e-4,
    train_batch_size=16,
    gener_batch_size=32,
    num_beams=1,
    print_predictions=False,
    fine_tune_first=True,
    n_epochs=2)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  if args.fine_tune_first:
    print('starting fine-tuning...')
    perform_fine_tuning()

  print('starting generation')
  perform_generation()
