#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, glob, re

def main():
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

  print(section_lengths)

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  note_dir = os.path.join(base, 'Thyme/Text/train/')

  main()
