#!/usr/bin/env python3

import sys, re, glob, argparse, shutil, os, random, numpy
from collections import defaultdict

sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import anafora
from transformers import T5Tokenizer
from dataset_base import ThymeDataset

# skip sections defined in eval/THYMEData.java
sections_to_skip = {'20104', '20105', '20116', '20138'}

def insert_at_offsets(text, offset2string):
  """Insert strings at specific offset"""

  dict_as_list = list(offset2string.items())
  dict_as_list.sort(key=lambda t: t[0], reverse=True)

  for offset, s in dict_as_list:
    text = text[:offset] + s + text[offset:]

  return text

def add_annotations(annot_tuples, ref_data, annot_type):
  """Add (span, id) tuples of anafora annotations to annot_tuples"""

  for annot in ref_data.annotations.select_type(annot_type):
    annot_begin, annot_end = annot.spans[0]
    annot_tuples.append((annot_begin, annot_end, annot.id))

def copy_annotations(from_data, to_data, annot_type):
  """Copy id, spans, and type of an annotation of specific type"""

  for annot in from_data.annotations.select_type(annot_type):
    entity = anafora.AnaforaEntity()
    entity.id = annot.id
    entity.spans = annot.spans
    entity.type = annot.type
    to_data.annotations.append(entity)

class Data(ThymeDataset):
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    out_dir,
    xml_regex,
    tokenizer,
    chunk_size,
    max_input_length,
    max_output_length):
    """Constructor"""

    super(Data, self).__init__(
      tokenizer,
      max_input_length,
      max_output_length)

    self.chunk_size = chunk_size
    self.xml_dir = xml_dir
    self.text_dir = text_dir
    self.out_dir = out_dir
    self.xml_regex = xml_regex

    # debug how many fall outside chunks
    self.captured_relations = []

    # count inputs/outputs that are too long
    self.in_over_maxlen = 0
    self.out_over_maxlen = 0

    # key: note path, value: (source, target) tuples
    self.note2rels = defaultdict(list)

    # key: note path; value: time expresions
    self.note2times = defaultdict(list)

    # key: note path; value: events
    self.note2events = defaultdict(list)

    # map t5 i/o instances to annotation offsets
    self.model_inputs_and_outputs()

  def chunk_generator(self, note_text):
    """Yield note chunk offsets of suitable length"""

    parag_re = r'(.+?\n)'
    sec_re = r'\[start section id=\"(.+)"\](.*?)\[end section id=\"\1"\]'

    # iterate over sections
    for sec_match in re.finditer(sec_re, note_text, re.DOTALL):

      section_id = sec_match.group(1)
      if section_id in sections_to_skip:
        continue

      section_text = sec_match.group(2)
      sec_start, sec_end = sec_match.start(2), sec_match.end(2)
      section_tokenized = self.tokenizer(section_text).input_ids

      # do we need to break this section into chunks?
      if len(section_tokenized) < self.chunk_size:
        yield sec_start, sec_end

      else:
        parag_offsets = []
        for parag_match in re.finditer(parag_re, section_text, re.DOTALL):
          parag_start, parag_end = parag_match.start(1), parag_match.end(1)
          parag_offsets.append((parag_start, parag_end))

        # form this many chunks (add an overflow chunk)
        n_chunks = (len(section_tokenized) // self.chunk_size) + 1

        for parags in numpy.array_split(parag_offsets, n_chunks):

          # this happens if there are fewer paragraphs than chunks
          # e.g. 2 large paragraphs in section and n_chunks is 3
          if parags.size == 0:
            continue

          chunk_start, _ = parags[0].tolist()
          _, chunk_end = parags[-1].tolist()
          yield sec_start + chunk_start, sec_start + chunk_end

  def notes_to_annotations(self):
    """Map note paths to relation, time, and event offsets"""

    for sub_dir, text_name, file_names in anafora.walk(self.xml_dir, self.xml_regex):
      note_path = os.path.join(self.text_dir, text_name)
      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      # populate (annot_start, annot_end, annot_id) tuples for a note
      add_annotations(self.note2times[note_path], ref_data, 'TIMEX3')
      add_annotations(self.note2times[note_path], ref_data, 'SECTIONTIME')
      add_annotations(self.note2times[note_path], ref_data, 'DOCTIME')
      add_annotations(self.note2events[note_path], ref_data, 'EVENT')

      # populate (src span, targ spans, src id, targ id) tuples for a note
      for rel in ref_data.annotations.select_type('TLINK'):
        src = rel.properties['Source']
        targ = rel.properties['Target']
        label = rel.properties['Type']
        if label == 'CONTAINS':
          self.note2rels[note_path].append(
            (src.spans[0], targ.spans[0], src.id, targ.id))

  def model_inputs_and_outputs(self):
    """Prepare i/o pairs to feed to T5"""

    # map note paths to annotation offsets
    self.notes_to_annotations()

    # count relation instances
    total_rel_count = 0

    for note_path in glob.glob(self.text_dir + 'ID*_clinic_*'):

      # some notes weren't annotated
      if note_path not in self.note2rels:
        continue

      # may be broken down into chunks later
      note_text = open(note_path).read()

      # iterate over note chunks
      for chunk_start, chunk_end in self.chunk_generator(note_text):

        # each event/time gets a number
        entity_num = 0

        # assign an id to each event and time
        time_offsets2int = {}
        event_offsets2int = {}

        # t5 i/o
        metadata = []
        rels_in_chunk = []

        for time_start, time_end, time_id in self.note2times[note_path]:
          if time_start >= chunk_start and time_end <= chunk_end:
            time_text = note_text[time_start:time_end]
            time_offsets2int[(time_start, time_end)] = entity_num
            metadata.append('%s/%s|%s' % (time_text, entity_num, time_id))
            entity_num += 1

        for event_start, event_end, event_id in self.note2events[note_path]:
          if event_start >= chunk_start and event_end <= chunk_end:
            event_text = note_text[event_start:event_end]
            event_offsets2int[(event_start, event_end)] = entity_num
            metadata.append('%s/%s|%s' % (event_text, entity_num, event_id))
            entity_num += 1

        for src_spans, targ_spans, src_id, targ_id in self.note2rels[note_path]:
          src_start, src_end = src_spans
          targ_start, targ_end = targ_spans

          # are both rel args inside this chunk?
          if src_start >= chunk_start and src_end <= chunk_end and \
             targ_start >= chunk_start and targ_end <= chunk_end:

            total_rel_count += 1
            self.captured_relations.append((src_id, targ_id))

            # retrieve id of the source argument
            if (src_start, src_end) in time_offsets2int:
              src_seq_num = time_offsets2int[(src_start, src_end)]
            else:
              src_seq_num = event_offsets2int[(src_start, src_end)]

            # retrieved id of the target argument
            if (targ_start, targ_end) in time_offsets2int:
              targ_seq_num = time_offsets2int[(targ_start, targ_end)]
            else:
              targ_seq_num = event_offsets2int[(targ_start, targ_end)]

            src = '%s/%s' % (note_text[src_start:src_end], src_seq_num)
            targ = '%s/%s' % (note_text[targ_start:targ_end], targ_seq_num)
            rels_in_chunk.append('CONTAINS(%s; %s)' % (src, targ))

        # add a seq num to all events/times in chunk text
        offset2str = {}
        for (start, end), entity_num in time_offsets2int.items():
          offset2str[end - chunk_start] = '/' + str(entity_num)
        for (start, end), entity_num in event_offsets2int.items():
          offset2str[end - chunk_start] = '/' + str(entity_num)
        chunk_text_with_markers = insert_at_offsets(
          note_text[chunk_start:chunk_end],
          offset2str)

        # mark event start and event end </e>

        metadata_str = '||'.join(metadata)
        input_str = 'task: RELEXT; text: %s' % chunk_text_with_markers
        if len(rels_in_chunk) > 0:
          output_str = ' '.join(rels_in_chunk)
        else:
          output_str = 'no relations found'

        # counts inputs and outputs that t5 cannot handle
        if len(self.tokenizer(input_str).input_ids) > self.max_input_length:
          self.in_over_maxlen += 1
        if len(self.tokenizer(output_str).input_ids) > self.max_input_length:
          self.in_over_maxlen += 1

        self.inputs.append(input_str)
        self.outputs.append(output_str)
        self.metadata.append(metadata_str)

    print('%d total input/output pairs' % len(self.inputs))
    print('%d total relation instances' % total_rel_count)
    print('%d inputs over maxlen' % self.in_over_maxlen)
    print('%d outputs over maxlen' % self.out_over_maxlen)

  def write_xml(self, predicted_relations):
    """Write predictions in anafora XML format"""

    # make a directory to write anafora xml
    if os.path.isdir(self.out_dir):
      shutil.rmtree(self.out_dir)
    os.mkdir(self.out_dir)

    # key: note, value: list of rel arg tuples
    note2rels = defaultdict(list)

    # map notes to relations in these notes
    for container_id, contained_id in predicted_relations:
      note_name = container_id.split('@')[2]
      note2rels[note_name].append((container_id, contained_id))

    # iterate over reference anafora xml files
    for sub_dir, text_name, file_names in anafora.walk(self.xml_dir, self.xml_regex):

      path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(path)

      # make a new XML file
      generated_data = anafora.AnaforaData()

      # copy gold events and times
      copy_annotations(ref_data, generated_data, 'EVENT')
      copy_annotations(ref_data, generated_data, 'TIMEX3')
      copy_annotations(ref_data, generated_data, 'SECTIONTIME')
      copy_annotations(ref_data, generated_data, 'DOCTIME')

      # add generated relations
      note_name = file_names[0].split('.')[0]
      for container_id, contained_id in note2rels[note_name]:
        relation = anafora.AnaforaRelation()
        relation.id = str(random.random())[2:]
        relation.type = 'TLINK'
        relation.parents_type = 'TemporalRelations'
        relation.properties['Source'] = container_id
        relation.properties['Type'] = 'CONTAINS'
        relation.properties['Target'] = contained_id
        generated_data.annotations.append(relation)

      # write xml to file
      generated_data.indent()
      os.mkdir(os.path.join(self.out_dir, sub_dir))
      out_path = os.path.join(self.out_dir, sub_dir, file_names[0])
      generated_data.to_file(out_path)

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xml_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Dev/'),
    text_dir = os.path.join(base, 'Thyme/Text/dev/'),
    xml_regex='.*[.]Temporal.*[.]xml',
    xml_out_dir='./Xml/',
    model_dir='Model/',
    model_name='t5-small',
    chunk_size=250,
    max_input_length=512,
    max_output_length=512)
  args = argparse.Namespace(**arg_dict)

  tokenizer = T5Tokenizer.from_pretrained(args.model_name)
  tokenizer.add_tokens(['CONTAINS'])

  rel_data = Data(
    xml_dir=args.xml_dir,
    text_dir=args.text_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length)
  rel_data.write_xml(rel_data.captured_relations)

  index = 6
  print('T5 INPUT:', rel_data.inputs[index] + '\n')
  print('T5 OUTPUT:', rel_data.outputs[index] + '\n')
  print('T5 METADATA:', rel_data.metadata[index])

  # note_path = os.path.join(args.text_dir, 'ID133_clinic_390')
  # note_text = open(note_path).read()
  # for start, end in rel_data.note_chunk_generator(note_text):
  #   print(note_text[start:end])
  #   print('='*30)

  # text = 'one two three four five six seven eight nine'
  # offset2string = {2:'1', 6:'2', 17:'3', 26:'4'}
  #
  # insert_at_offsets(text, offset2string)
