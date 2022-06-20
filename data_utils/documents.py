from typing import *

import re
import cv2
import json
import string
from pathlib import Path

from torchtext.data import Field, RawField
import numpy as np

from utils.entities_list import Entities_list
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls, entities_vocab_cls

MAXIMUM_BOX = 65
TRANS_LEN = 35


TextSegmentsField = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True)
TextSegmentsField.vocab = keys_vocab_cls

IOBTagsField = Field(sequential=True, is_target=True, use_vocab=True, batch_first=True)
IOBTagsField.vocab = iob_labels_vocab_cls


class Document:

    def __init__(self, boxes_and_transcripts_file: Path, image_file: Path,
                 resized_image_size: Tuple[int, int] = (480, 960),
                 iob_tagging_type: str = 'box_level', entities_file: Path = None, training: bool = True,
                 image_index=None):

        self.resized_image_size = resized_image_size

        self.training = training

        assert iob_tagging_type in ['box_level', 'document_level', 'box_and_within_box_level'], \
            'iob tagging type {} is not supported'.format(iob_tagging_type)
        self.iob_tagging_type = iob_tagging_type

        try:

            if self.training:
                boxes_and_transcripts_data = read_gt_file_with_box_entity_type(boxes_and_transcripts_file.as_posix())
            else:
                boxes_and_transcripts_data = read_ocr_file_without_box_entity_type(
                    boxes_and_transcripts_file.as_posix())

            boxes_and_transcripts_data = sort_box_with_list(boxes_and_transcripts_data)

            image = cv2.imread(image_file.as_posix())
        except Exception as e:
            raise IOError('Error occurs in image {}: {}'.format(image_file.stem, e.args))

        boxes, transcripts, box_entity_types = [], [], []
        if self.training:
            
            for index, points, transcript, entity_type in boxes_and_transcripts_data:
                if len(transcript) == 0:
                    
                    transcript = ' '
                boxes.append(points)
                transcripts.append(transcript)
                box_entity_types.append(entity_type)
        else:
            
            for index, points, transcript in boxes_and_transcripts_data:
                if len(transcript) == 0:
                    
                    transcript = ' '
                boxes.append(points)
                transcripts.append(transcript)

        boxes_num = min(len(boxes), MAXIMUM_BOX)
        transcript_len = min(max([len(t) for t in transcripts[:boxes_num]]), TRANS_LEN)
        mask = np.zeros((boxes_num, transcript_len), dtype=int)

        relation_features = np.zeros((boxes_num, boxes_num, 6))

        try:

            height, width, _ = image.shape

            image = cv2.resize(image, self.resized_image_size, interpolation=cv2.INTER_LINEAR)
            x_scale = self.resized_image_size[0] / width
            y_scale = self.resized_image_size[1] / height

            min_area_boxes = [cv2.minAreaRect(np.array(box, dtype=np.float32).reshape(4, 2)) for box in
                              boxes[:boxes_num]]

            resized_boxes = []
            for i in range(boxes_num):
                box_i = boxes[i]
                transcript_i = transcripts[i]

                resized_box_i = [int(np.round(pos * x_scale)) if i % 2 == 0 else int(np.round(pos * y_scale))
                                 for i, pos in enumerate(box_i)]

                resized_rect_output_i = cv2.minAreaRect(np.array(resized_box_i, dtype=np.float32).reshape(4, 2))
                resized_box_i = cv2.boxPoints(resized_rect_output_i)
                resized_box_i = resized_box_i.reshape((8,))
                resized_boxes.append(resized_box_i)

                self.relation_features_between_ij_nodes(boxes_num, i, min_area_boxes, relation_features, transcript_i,
                                                        transcripts)

            relation_features = normalize_relation_features(relation_features, width=width, height=height)
            text_segments = [list(trans) for trans in transcripts[:boxes_num]]

            if self.training:

                if self.iob_tagging_type != 'box_level':
                    with entities_file.open() as f:
                        entities = json.load(f)

                if self.iob_tagging_type == 'box_level':

                    iob_tags_label = text2iob_label_with_box_level_match(box_entity_types[:boxes_num],
                                                                         transcripts[:boxes_num])
                elif self.iob_tagging_type == 'document_level':

                    iob_tags_label = text2iob_label_with_document_level_exactly_match(transcripts[:boxes_num], entities)
                elif self.iob_tagging_type == 'box_and_within_box_level':

                    iob_tags_label = iob_con(box_entity_types[:boxes_num],
                                                                                          transcripts[:boxes_num],
                                                                                          entities, ['address'])

                iob_tags_label = IOBTagsField.process(iob_tags_label)[:, :transcript_len].numpy()
                box_entity_types = [entities_vocab_cls.stoi[t] for t in box_entity_types[:boxes_num]]

            texts, texts_len = TextSegmentsField.process(text_segments)
            texts = texts[:, :transcript_len].numpy()
            texts_len = np.clip(texts_len.numpy(), 0, transcript_len)
            text_segments = (texts, texts_len)

            for i in range(boxes_num):
                mask[i, :texts_len[i]] = 1

            self.whole_image = RawField().preprocess(image)
            
            self.text_segments = TextSegmentsField.preprocess(text_segments)
            
            self.boxes_coordinate = RawField().preprocess(resized_boxes)
            
            self.relation_features = RawField().preprocess(relation_features)
            
            self.mask = RawField().preprocess(mask)
            
            self.boxes_num = RawField().preprocess(boxes_num)
            
            self.transcript_len = RawField().preprocess(transcript_len)
            
            if self.training:
                self.iob_tags_label = IOBTagsField.preprocess(iob_tags_label)
            else:
                self.image_index = RawField().preprocess(image_index)

        except Exception as e:
            raise RuntimeError('Error occurs  {}: {}'.format(boxes_and_transcripts_file.stem, e.args))

    def relation_features_between_ij_nodes(self, boxes_num, i, min_area_boxes, relation_features, transcript_i,
                                           transcripts):

        for j in range(boxes_num):
            transcript_j = transcripts[j]

            rect_output_i = min_area_boxes[i]
            rect_output_j = min_area_boxes[j]

            center_i = rect_output_i[0]
            center_j = rect_output_j[0]

            width_i, height_i = rect_output_i[1]
            width_j, height_j = rect_output_j[1]

            relation_features[i, j, 0] = np.abs(center_i[0] - center_j[0]) \
                if np.abs(center_i[0] - center_j[0]) is not None else -1 

            relation_features[i, j, 1] = np.abs(center_i[1] - center_j[1]) \
                if np.abs(center_i[1] - center_j[1]) is not None else -1  

            relation_features[i, j, 2] = width_i / (height_i) \
                if width_i / (height_i) is not None else -1  

            relation_features[i, j, 3] = height_j / (height_i) \
                if height_j / (height_i) is not None else -1  
            relation_features[i, j, 4] = width_j / (height_i) \
                if width_j / (height_i) is not None else -1  
            relation_features[i, j, 5] = len(transcript_j) / (len(transcript_i)) \
                if len(transcript_j) / (len(transcript_i)) is not None else -1 


def read_gt_file_with_box_entity_type(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        document_text = f.read()

    regex = r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,(.*)\n?$"

    matches = re.finditer(regex, document_text, re.MULTILINE)

    res = []
    for matchNum, match in enumerate(matches, start=1):
        index = int(match.group(1))
        points = [float(match.group(i)) for i in range(2, 10)]
        transcription = str(match.group(10))
        entity_type = str(match.group(11))
        res.append((index, points, transcription, entity_type))
    return res


def read_ocr_file_without_box_entity_type(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        document_text = f.read()

    regex = r"^\s*(-?\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$"

    matches = re.finditer(regex, document_text, re.MULTILINE)

    res = []
    for matchNum, match in enumerate(matches, start=1):
        index = int(match.group(1))
        points = [float(match.group(i)) for i in range(2, 10)]
        transcription = str(match.group(10))
        res.append((index, points, transcription))
    return res


def sort_box_with_list(data: List[Tuple], left_right_first=False):
    def compare_key(x):
        points = x[1]
        box = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]],
                       dtype=np.float32)
        rect = cv2.minAreaRect(box)
        center = rect[0]
        if left_right_first:
            return center[0], center[1]
        else:
            return center[1], center[0]

    data = sorted(data, key=compare_key)
    return data


def normalize_relation_features(feat: np.ndarray, width: int, height: int):
    np.clip(feat, 1e-8, np.inf)
    feat[:, :, 0] = feat[:, :, 0] / width
    feat[:, :, 1] = feat[:, :, 1] / height

    for i in range(2, 6):
        feat_ij = feat[:, :, i]
        max_value = np.max(feat_ij)
        min_value = np.min(feat_ij)
        if max_value != min_value:
            feat[:, :, i] = feat[:, :, i] - min_value / (max_value - min_value)
    return feat


def text2iob_label_with_box_level_match(annotation_box_types: List[str], transcripts: List[str]) -> List[List[str]]:

    tags = []
    for entity_type, transcript in zip(annotation_box_types, transcripts):
        if entity_type in Entities_list:
            if len(transcript) == 1:
                tags.append(['B-{}'.format(entity_type)])
            else:
                tag = ['I-{}'.format(entity_type)] * len(transcript)
                tag[0] = 'B-{}'.format(entity_type)
                tags.append(tag)
        else:
            tags.append(['O'] * len(transcript))

    return tags


def text2iob_label_with_document_level_exactly_match(transcripts: List[str], exactly_entities_label: Dict) -> List[
        List[str]]:

    concatenated_sequences = []
    sequences_len = []
    for transcript in transcripts:
        concatenated_sequences.extend(list(transcript))
        sequences_len.append(len(transcript))

    result_tags = ['O'] * len(concatenated_sequences)
    for entity_type, entity_value in exactly_entities_label.items():
        if entity_type not in Entities_list:
            continue
        (src_seq, src_idx), (tgt_seq, _) = tranfer(concatenated_sequences), tranfer(
            entity_value)
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
            continue

        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i + tgt_len] == tgt_seq:
                tag = ['I-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                tag[0] = 'B-{}'.format(entity_type)
                result_tags[src_idx[i]:src_idx[i + tgt_len - 1] + 1] = tag

    tagged_transcript = []
    start = 0
    for length in sequences_len:
        tagged_transcript.append(result_tags[start: start + length])
        start = start + length
        if start >= len(result_tags):
            break
    return tagged_transcript


def iob_con(annotation_box_types: List[str],
                                                         transcripts: List[str],
                                                         exactly_entities_label: Dict[str, str],
                                                         box_level_entities: List[str]) -> List[List[str]]:

    def exactly_match_within_box(transcript: str, entity_type: str, entity_exactly_value: str):

        matched = False
        (src_seq, src_idx), (tgt_seq, _) = tranfer(transcript), tranfer(
            entity_exactly_value)
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
            return matched, None

        result_tags = ['O'] * len(transcript)
        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i + tgt_len] == tgt_seq:
                matched = True
                tag = ['I-{}'.format(entity_type)] * (src_idx[i + tgt_len - 1] - src_idx[i] + 1)
                tag[0] = 'B-{}'.format(entity_type)
                result_tags[src_idx[i]:src_idx[i + tgt_len - 1] + 1] = tag
                break

        return matched, result_tags

    tags = []
    for entity_type, transcript in zip(annotation_box_types, transcripts):
        entity_type = entity_type.strip()
        if entity_type in Entities_list:

            matched, resulted_tag = False, None
            if entity_type not in box_level_entities:
                matched, resulted_tag = exactly_match_within_box(transcript, entity_type,
                                                                 exactly_entities_label[entity_type])

            if matched:
                tags.append(resulted_tag)
            else:
                tag = ['I-{}'.format(entity_type)] * len(transcript)
                tag[0] = 'B-{}'.format(entity_type)
                tags.append(tag)
        else:
            tags.append(['O'] * len(transcript))

    return tags


def tranfer(transcripts: List[str]):

    seq, idx = [], []
    for index, x in enumerate(transcripts):
        if x not in string.punctuation and x not in string.whitespace:
            seq.append(x)
            idx.append(index)
    return seq, idx

