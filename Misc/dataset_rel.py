#!/usr/bin/env python3

import sys, re, glob

sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os
import anafora

class RelData:
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    xml_regex):
    """Constructor"""

    self.xml_dir = xml_dir
    self.text_dir = text_dir
    self.xml_regex = xml_regex

    self.map_notes_to_relations()
    self.map_sections_to_relations()

  def map_sections_to_relations(self):
    """Sectionize and index"""

    for note_path in glob.glob(self.text_dir + 'ID*'):

      # some notes weren't annotated
      if note_path not in self.note2args:
        continue

      print(note_path)
      note_text = open(note_path).read()
      regex_str = r'\[start section id=\"(.+)"\](.*?)\[end section id=\"\1"\]'

      for match in re.finditer(regex_str, note_text, re.DOTALL):

        # section offsets
        sec_start, sec_end = match.start(2), match.end(2)
        print(note_text[sec_start:sec_end])

        # look for relation arguments contained in this section
        for src_spans, targ_spans in self.note2args[note_path]:
          src_start, src_end = src_spans
          targ_start, targ_end = targ_spans
          if src_start >= sec_start and src_end <= sec_end:
            src = note_text[src_start:src_end]
            targ = note_text[targ_start:targ_end]
            print('contains(%s, %s)' % (src, targ))

  def map_notes_to_relations(self):
    """Make x, y etc."""

    # key: note path, value: (source, target) tuples
    self.note2args = {}

    for sub_dir, text_name, file_names in anafora.walk(self.xml_dir, self.xml_regex):
      text_path = os.path.join(self.text_dir, text_name)
      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      rel_args = []
      for rel in ref_data.annotations.select_type('TLINK'):
        source = rel.properties['Source']
        target = rel.properties['Target']
        label = rel.properties['Type']

        if label == 'CONTAINS':
          rel_args.append((source.spans[0], target.spans[0]))

      self.note2args[text_path] = rel_args

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']

  rel_data = RelData(
    xml_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Train/'),
    text_dir=os.path.join(base, 'Thyme/Text/train/'),
    xml_regex='.*[.]Temporal.*[.]xml')