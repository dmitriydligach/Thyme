#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import random, argparse, os, shutil, importlib, torch, re

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

  # relation arg id tuples
  pred_rels = []

  for batch in data_loader:

    # metadata for this batch
    metadata = batch.pop('metadata')

    for key in batch.keys():
      batch[key] = batch[key].to(device)

    # generated tensor: (batch_size, max_output_length)
    predictions = model.generate(
      do_sample=True,
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

    # time and event metadata example for a section
    # February 8, 2010|379@e@ID128_clinic_377@gold||currently|388@e@ID128_clinic_377@gold
    # anastomosis|334@e@ID128_clinic_377@gold||diagnosis|336@e@ID128_clinic_377@gold

    # iterate over samples in this batch
    for i in range(len(predictions)):
      if args.print_predictions:
        print('[input]', inputs[i], '\n')
        print('[targets]', targets[i], '\n')
        print('[predict]', predictions[i], '\n')
        print('[metadata]', metadata[i], '\n\n')

      # match argument text in predictions
      # CONTAINS(February 8, 2010; scan) CONTAINS(currently; denies)
      regex_str = r'CONTAINS\((.+?); (.+?)\)'
      matched_args = re.findall(regex_str, predictions[i], re.DOTALL)
      if len(matched_args) == 0:
        # no relations generated
        continue

      # map arg text to ids
      if len(metadata[i]) == 0:
        # todo: what if t5 still generates something?
        # no events or times in this section
        continue

      # parse metadata and map args to anafora ids
      arg_text2id = {}
      for arg_id_pair in metadata[i].split('||'):
        elements = arg_id_pair.split('|')
        if len(elements) == 2:
          arg_text, arg_id = elements
          arg_text2id[arg_text] = arg_id

      # convert generated relations to anafora id pairs
      for arg1, arg2 in matched_args:
        if arg1 in arg_text2id and arg2 in arg_text2id:
          pred_rels.append((arg_text2id[arg1], arg_text2id[arg2]))
        else:
          # t5 generated a rel over non-gold events/times
          pass

  return pred_rels

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
  tokenizer.add_tokens(['<t>', '</t>', '<e>', '</e>'])
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
  train_data_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=args.train_batch_size)

  test_dataset = data.Data(
    xml_dir=args.xml_test_dir,
    text_dir=args.text_test_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)
  val_data_loader = DataLoader(
    test_dataset,
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
  tokenizer.add_tokens(['<t>', '</t>', '<e>', '</e>'])
  model.resize_token_embeddings(len(tokenizer)) # todo: need this?

  test_dataset = data.Data(
    xml_dir=args.xml_test_dir,
    text_dir=args.text_test_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)
  test_data_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=args.gener_batch_size)

  # generate output from the saved model
  predicted_relations = generate(model, test_data_loader, tokenizer)
  print('writing xml...')
  test_dataset.write_xml(predicted_relations)

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
    chunk_size=100,
    max_input_length=400,
    max_output_length=400,
    n_files='all',
    learning_rate=1e-3,
    train_batch_size=16,
    gener_batch_size=16,
    num_beams=3,
    print_predictions=False,
    do_train=True,
    n_epochs=5)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  if args.do_train:
    print('starting training...')
    perform_fine_tuning()

  print('starting generation...')
  perform_generation()
