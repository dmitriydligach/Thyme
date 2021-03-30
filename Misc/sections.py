#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, glob, re, numpy
import matplotlib.pyplot as plt
from transformers import T5Tokenizer

def split_on_spaces():
  """Look at section length"""

  # match sections that look as follows:
  # [start section id="20101"]
  # [end section id="20111"]
  # [start section id="$final"]

  section_lengths = []

  for note_name in glob.glob(note_dir + 'ID*'):
    note_text = open(note_name).read()
    sections = re.split(r'\[\w+ section id=\".+\"\]', note_text)

    for section in sections:
      tokens = section.split()
      if len(tokens) > 5:
        section_lengths.append(len(tokens))

  print('total notes:', len(glob.glob(note_dir + 'ID*')))
  print('total sections:', len(section_lengths))
  print('mean:', numpy.mean(section_lengths))
  print('median:', numpy.median(section_lengths))
  print('std:', numpy.std(section_lengths))
  print('max:', numpy.max(section_lengths))
  print('min:', numpy.min(section_lengths))

  plt.hist(x=section_lengths, bins='auto')
  plt.xlabel('Section length')
  plt.ylabel('Token count')
  plt.title('Section length histogram')
  plt.show()

def split_using_tokenizer():
  """Look at section length"""

  tokenizer = T5Tokenizer.from_pretrained('t5-small')

  # match sections that look as follows:
  # [start section id="20101"]
  # [end section id="20111"]
  # [start section id="$final"]

  section_lengths = []

  for note_name in glob.glob(note_dir + 'ID*'):
    note_text = open(note_name).read()
    sections = re.split(r'\[\w+ section id=\".+\"\]', note_text)

    for section in sections:
      tokens = tokenizer(section).input_ids
      if len(tokens) > 5:
        section_lengths.append(len(tokens))

  print('total notes:', len(glob.glob(note_dir + 'ID*')))
  print('total sections:', len(section_lengths))
  print('mean:', numpy.mean(section_lengths))
  print('median:', numpy.median(section_lengths))
  print('std:', numpy.std(section_lengths))
  print('max:', numpy.max(section_lengths))
  print('min:', numpy.min(section_lengths))

  over = numpy.where(numpy.array(section_lengths) > 512)
  print('above 512:', len(over[0]))

  plt.hist(x=section_lengths, bins=25)
  plt.xlabel('Section length')
  plt.ylabel('Token count')
  plt.title('Section length histogram')
  # plt.show()
  plt.savefig('books_read.png')

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  note_dir = os.path.join(base, 'Thyme/Text/train/')

  # split_on_spaces()
  split_using_tokenizer()
