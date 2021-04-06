#!/usr/bin/env python3

import sys, re, glob, argparse

sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os
import anafora
from transformers import T5Tokenizer
from dataset_base import ThymeDataset

class Data(ThymeDataset):
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    xml_regex,
    tokenizer,
    max_input_length,
    max_output_length):
    """Constructor"""

    super(Data, self).__init__(
      tokenizer,
      max_input_length,
      max_output_length)

    self.xml_dir = xml_dir
    self.text_dir = text_dir
    self.xml_regex = xml_regex

    self.map_notes_to_relations()
    self.map_sections_to_relations()

  def map_notes_to_relations(self):
    """Map note paths to relation argument offsets"""

    # key: note path, value: (source, target) tuples
    self.note2args = {}

    for sub_dir, text_name, file_names in anafora.walk(self.xml_dir, self.xml_regex):
      note_path = os.path.join(self.text_dir, text_name)
      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      # (src_start, src_end, targ_start, targ_end) tuples
      rel_args = []

      for rel in ref_data.annotations.select_type('TLINK'):
        source = rel.properties['Source']
        target = rel.properties['Target']
        label = rel.properties['Type']

        if label == 'CONTAINS':
          rel_args.append((source.spans[0], target.spans[0]))

      self.note2args[note_path] = rel_args

  def map_sections_to_relations(self):
    """Sectionize and index"""

    # iterate over notes and sectionize them
    for note_path in glob.glob(self.text_dir + 'ID*'):

      # some notes weren't annotated
      if note_path not in self.note2args:
        continue

      note_text = open(note_path).read()
      regex_str = r'\[start section id=\"(.+)"\](.*?)\[end section id=\"\1"\]'

      # iterate over sections
      for match in re.finditer(regex_str, note_text, re.DOTALL):
        sec_start, sec_end = match.start(2), match.end(2)

        rels_in_section = []
        for src_spans, targ_spans in self.note2args[note_path]:
          src_start, src_end = src_spans
          targ_start, targ_end = targ_spans

          if src_start >= sec_start and src_end <= sec_end and \
             targ_start >= sec_start and targ_end <= sec_end:
            src = note_text[src_start:src_end]
            targ = note_text[targ_start:targ_end]
            rels_in_section.append('CONTAINS(%s, %s)' % (src, targ))

        self.inputs.append(match.group(2))
        self.outputs.append(' '.join(rels_in_section))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xml_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Train/'),
    text_dir = os.path.join(base, 'Thyme/Text/train/'),
    xml_regex='.*[.]Temporal.*[.]xml',
    xml_out_dir='./Xml/',
    model_dir='Model/',
    model_name='t5-small',
    max_input_length=512,
    max_output_length=512)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters:', args)

  tokenizer = T5Tokenizer.from_pretrained(args.model_name)
  tokenizer.add_tokens(['<e>', '</e>'])

  rel_data = Data(
    xml_dir=args.xml_dir,
    text_dir=args.text_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)

  print(rel_data.inputs[4])
  print(rel_data.outputs[4])
