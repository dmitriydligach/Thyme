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

    # key: note path, value: (source, target) tuples
    self.note2args = {}

    # key: note path; value: time expresions
    self.note2times = {}

    # key: note path; value: events
    self.note2events = {}

    self.map_notes_to_annotations()
    self.map_sections_to_annotations()

  def map_notes_to_annotations(self):
    """Map note paths to relation, time, and event offsets"""

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

      # (time_start, time_end, time_id) tuples
      times = []
      for time in ref_data.annotations.select_type('TIMEX3'):
        time_begin, time_end = time.spans[0]
        times.append((time_begin, time_end, time.id))
      self.note2times[note_path] = times

      # (event_start, event_end, event_id) tuples
      events = []
      for event in ref_data.annotations.select_type('EVENT'):
        event_begin, event_end = event.spans[0]
        events.append((event_begin, event_end, event.id))
      self.note2events[note_path] = events

  def map_sections_to_annotations(self):
    """Sectionize and index"""

    # todo: figure out what sections to skip

    # iterate over notes and sectionize them
    for note_path in glob.glob(self.text_dir + 'ID*'):

      # some notes weren't annotated
      if note_path not in self.note2args:
        continue

      note_text = open(note_path).read()
      regex_str = r'\[start section id=\"(.+)"\](.*?)\[end section id=\"\1"\]'

      # iterate over sections
      for match in re.finditer(regex_str, note_text, re.DOTALL):
        section_text = match.group(2)
        sec_start, sec_end = match.start(2), match.end(2)

        rels_in_sec = []
        for src_spans, targ_spans in self.note2args[note_path]:
          src_start, src_end = src_spans
          targ_start, targ_end = targ_spans
          if src_start >= sec_start and src_end <= sec_end and \
             targ_start >= sec_start and targ_end <= sec_end:
            src = note_text[src_start:src_end]
            targ = note_text[targ_start:targ_end]
            rels_in_sec.append('CONTAINS(%s; %s)' % (src, targ))

        times_in_sec = []
        time_metadata = []
        for time_start, time_end, time_id in self.note2times[note_path]:
          if time_start >= sec_start and time_end <= sec_end:
            time_text = note_text[time_start:time_end]
            times_in_sec.append(time_text)
            time_metadata.append('%s|%s' % (time_text, time_id))

        events_in_sec = []
        event_metadata = []
        for event_start, event_end, event_id in self.note2events[note_path]:
          if event_start >= sec_start and event_end <= sec_end:
            event_text = note_text[event_start:event_end]
            events_in_sec.append(event_text)
            event_metadata.append('%s|%s' % (event_text, event_id))

        input_str = 'task: REL; section: %s; events: %s; times: %s' % \
          (section_text, ', '.join(events_in_sec), ', '.join(times_in_sec))

        if len(rels_in_sec) > 0:
          output_str = ' '.join(rels_in_sec)
        else:
          output_str = 'no relations found'

        time_metadata_str = '||'.join(time_metadata)
        event_metadata_str = '||'.join(event_metadata)

        self.inputs.append(input_str)
        self.outputs.append(output_str)
        self.time_metadata.append(time_metadata_str)
        self.event_metadata.append(event_metadata_str)

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

  tokenizer = T5Tokenizer.from_pretrained(args.model_name)
  # tokenizer.add_tokens(['<e>', '</e>'])

  rel_data = Data(
    xml_dir=args.xml_dir,
    text_dir=args.text_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)

  index = 4
  print('T5 INPUT:', rel_data.inputs[index])
  print('T5 OUTPUT:', rel_data.outputs[index])
  print('T5 TIME METADATA:', rel_data.time_metadata[index])
  print('T5 EVENT METADATA:', rel_data.event_metadata[index])
