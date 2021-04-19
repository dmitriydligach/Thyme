#!/usr/bin/env python3

import sys, re, glob, argparse, shutil, os, random
from collections import defaultdict

sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import anafora
from transformers import T5Tokenizer
from dataset_base import ThymeDataset

# ids of sections that weren't annotated
sections_to_skip = {'20104', '20105', '20116', '20138'}

class Data(ThymeDataset):
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    out_dir,
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
    self.out_dir = out_dir
    self.xml_regex = xml_regex

    # key: note path, value: (source, target) tuples
    self.note2args = {}

    # key: note path; value: time expresions
    self.note2times = {}

    # key: note path; value: events
    self.note2events = {}

    # note path mapped to annotation offsets
    self.map_notes_to_annotations()

    # t5 i/o instances mapped to annotation offsets
    self.map_sections_to_annotations()

  def map_notes_to_annotations(self):
    """Map note paths to relation, time, and event offsets"""

    for sub_dir, text_name, file_names in anafora.walk(self.xml_dir, self.xml_regex):
      note_path = os.path.join(self.text_dir, text_name)
      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

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

      # (src_start, src_end, targ_start, targ_end) tuples
      rel_args = []
      for rel in ref_data.annotations.select_type('TLINK'):
        source = rel.properties['Source']
        target = rel.properties['Target']
        label = rel.properties['Type']
        if label == 'CONTAINS':
          rel_args.append((source.spans[0], target.spans[0]))
      self.note2args[note_path] = rel_args

  def map_sections_to_annotations(self):
    """Sectionize and index"""

    # iterate over clinical notes and sectionize them
    for note_path in glob.glob(self.text_dir + 'ID*_clinic_*'):

      # some notes weren't annotated
      if note_path not in self.note2args:
        continue

      note_text = open(note_path).read()
      regex_str = r'\[start section id=\"(.+)"\](.*?)\[end section id=\"\1"\]'

      # iterate over sections
      for match in re.finditer(regex_str, note_text, re.DOTALL):

        section_id = match.group(1)
        if section_id in sections_to_skip:
          continue

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

        metadata = []
        times_in_sec = []
        for time_start, time_end, time_id in self.note2times[note_path]:
          if time_start >= sec_start and time_end <= sec_end:
            time_text = note_text[time_start:time_end]
            times_in_sec.append(time_text)
            metadata.append('%s|%s' % (time_text, time_id))

        events_in_sec = []
        for event_start, event_end, event_id in self.note2events[note_path]:
          if event_start >= sec_start and event_end <= sec_end:
            event_text = note_text[event_start:event_end]
            events_in_sec.append(event_text)
            metadata.append('%s|%s' % (event_text, event_id))

        metadata_str = '||'.join(metadata)
        input_str = 'task: REL; section: %s; events: %s; times: %s' % \
          (section_text, ', '.join(events_in_sec), ', '.join(times_in_sec))
        if len(rels_in_sec) > 0:
          output_str = ' '.join(rels_in_sec)
        else:
          output_str = 'no relations found'

        self.inputs.append(input_str)
        self.outputs.append(output_str)
        self.metadata.append(metadata_str)

  def write_xml(self, predicted_relations):
    """Write predictions in anafora XML format"""

    # make a directory to write anafora xml
    if os.path.isdir(self.out_dir):
      shutil.rmtree(self.out_dir)
    os.mkdir(self.out_dir)

    # key: note, value: list of rel arg tuples
    note2rels = defaultdict(list)

    for container_id, contained_id in predicted_relations:
      note_name = container_id.split('@')[2]
      note2rels[note_name].append((container_id, contained_id))

    # iterate over reference xml files
    for sub_dir, text_name, file_names in \
            anafora.walk(self.xml_dir, self.xml_regex):

      path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(path)

      # make a new XML file
      data = anafora.AnaforaData()

      # copy gold events
      for event in ref_data.annotations.select_type('EVENT'):
        entity = anafora.AnaforaEntity()
        entity.id = event.id
        entity.spans = event.spans
        entity.type = event.type
        data.annotations.append(entity)

      # copy gold time expressions
      for time in ref_data.annotations.select_type('TIMEX3'):
        entity = anafora.AnaforaEntity()
        entity.id = time.id
        entity.spans = time.spans
        entity.type = time.type
        data.annotations.append(entity)

      # add generated relations
      note_name = file_names[0].split('.')[0]
      for container_id, contained_id in note2rels[note_name]:
        relation = anafora.AnaforaRelation()
        relation.id = str(random.random())
        relation.type = 'TLINK'
        relation.parents_type = 'TemporalRelations'
        relation.properties['Source'] = container_id
        relation.properties['Type'] = 'CONTAINS'
        relation.properties['Target'] = contained_id
        data.annotations.append(relation)

      # write xml to file
      data.indent()
      os.mkdir(os.path.join(self.out_dir, sub_dir))
      out_path = os.path.join(self.out_dir, sub_dir, file_names[0])
      data.to_file(out_path)

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
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)

  index = 4
  print('T5 INPUT:', rel_data.inputs[index] + '\n')
  print('T5 OUTPUT:', rel_data.outputs[index] + '\n')
  print('T5 METADATA:', rel_data.metadata[index])

  # predicted_relations = (('75@e@ID077_clinic_229@gold', '74@e@ID077_clinic_229@gold'),
  #                        ('92@e@ID077_clinic_229@gold', '54@e@ID077_clinic_229@gold'),
  #                        ('142@e@ID021_clinic_063@gold', '213@e@ID021_clinic_063@gold'),
  #                        ('89@e@ID021_clinic_063@gold', '66@e@ID021_clinic_063@gold'))
  # rel_data.write_xml(predicted_relations)
