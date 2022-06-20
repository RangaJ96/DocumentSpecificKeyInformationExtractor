
from typing import *
from pathlib import Path
import warnings
import random
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from . import documents
from .documents import Document
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls, entities_vocab_cls


class Dataset(Dataset):

    def __init__(self, files_name: str = None,
                 boxes_and_transcripts_folder: str = 'boxes_and_transcripts',
                 images_folder: str = 'images',
                 entities_folder: str = 'entities',
                 iob_tagging_type: str = 'box_and_within_box_level',
                 resized_image_size: Tuple[int, int] = (480, 960),
                 keep_ratio: bool = True,
                 ignore_error: bool = False,
                 training: bool = True
                 ):
        
        super().__init__()

        self.iob_tagging_type = iob_tagging_type
        self.keep_ratio = keep_ratio
        self.ignore_error = ignore_error
        self.training = training
        assert resized_image_size and len(resized_image_size) == 2, 'resized image size not be set.'
        self.resized_image_size = tuple(resized_image_size) 

        if self.training: 
            self.files_name = Path(files_name)
            self.data_root = self.files_name.parent
            self.boxes_and_transcripts_folder: Path = self.data_root.joinpath(boxes_and_transcripts_folder)
            self.images_folder: Path = self.data_root.joinpath(images_folder)
            self.entities_folder: Path = self.data_root.joinpath(entities_folder)
            if self.iob_tagging_type != 'box_level':
                if not self.entities_folder.exists():
                    raise FileNotFoundError('Entity folder is not exist!')
        else:  
            self.boxes_and_transcripts_folder: Path = Path(boxes_and_transcripts_folder)
            self.images_folder: Path = Path(images_folder)
        print("boxes_and_transcripts_folder", boxes_and_transcripts_folder)
        if not (self.boxes_and_transcripts_folder.exists() and self.images_folder.exists()):
            raise FileNotFoundError('Not contain boxes_and_transcripts floader {} or images folder {}.'
                                    .format(self.boxes_and_transcripts_folder.as_posix(),
                                            self.images_folder.as_posix()))
        if self.training:
            self.files_list = pd.read_csv(self.files_name.as_posix(), header=None,
                                          names=['index', 'document_class', 'file_name'],
                                          dtype={'index': int, 'document_class': str, 'file_name': str})
        else:
            self.files_list = list(self.boxes_and_transcripts_folder.glob('*.tsv'))

    def __len__(self):
        return len(self.files_list)


    def __getitem__(self, index):

        if self.training:
            dataitem: pd.Series = self.files_list.iloc[index]
           
            boxes_and_transcripts_file = self.boxes_and_transcripts_folder.joinpath(
                Path(dataitem['file_name']).stem + '.tsv')
            image_file = self.images_folder.joinpath(Path(dataitem['file_name']).stem + '.jpg')
            entities_file = self.entities_folder.joinpath(Path(dataitem['file_name']).stem + '.txt')
           
        else:
            boxes_and_transcripts_file = self.boxes_and_transcripts_folder.joinpath(
                Path(self.files_list[index]).stem + '.tsv')
            image_file = self.images_folder.joinpath(Path(self.files_list[index]).stem + '.jpg')

        if not boxes_and_transcripts_file.exists() or not image_file.exists():
            if self.ignore_error and self.training:
                warnings.warn('{} is not exist. get a new one.'.format(boxes_and_transcripts_file))
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Sample: {} not exist.'.format(boxes_and_transcripts_file.stem))

        try:
            

            if self.training:
                document = documents.Document(boxes_and_transcripts_file, image_file, self.resized_image_size,
                                              self.iob_tagging_type, entities_file, training=self.training)
            else:
                document = documents.Document(boxes_and_transcripts_file, image_file, self.resized_image_size,
                                              image_index=index, training=self.training)
            return document
        except Exception as e:
            if self.ignore_error:
                warnings.warn('loading samples is occurring error, try to regenerate a new one.')
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError('Error occurs  {}: {}'.format(boxes_and_transcripts_file.stem, e.args))


class BatchCollateFn(object):

    def __init__(self, training: bool = True):
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.training = training

    def __call__(self, batch_list: List[Document]):

       
        max_boxes_num_batch = max([x.boxes_num for x in batch_list])
        
        max_transcript_len = max([x.transcript_len for x in batch_list])

        
        
        image_batch_tensor = torch.stack([self.trsfm(x.whole_image) for x in batch_list], dim=0).float()

        
        relation_features_padded_list = [F.pad(torch.FloatTensor(x.relation_features),
                                               (0, 0, 0, max_boxes_num_batch - x.boxes_num,
                                                0, max_boxes_num_batch - x.boxes_num))
                                         for i, x in enumerate(batch_list)]
        relation_features_batch_tensor = torch.stack(relation_features_padded_list, dim=0)

        
        boxes_coordinate_padded_list = [F.pad(torch.FloatTensor(x.boxes_coordinate),
                                              (0, 0, 0, max_boxes_num_batch - x.boxes_num))
                                        for i, x in enumerate(batch_list)]
        boxes_coordinate_batch_tensor = torch.stack(boxes_coordinate_padded_list, dim=0)

        
        text_segments_padded_list = [F.pad(torch.LongTensor(x.text_segments[0]),
                                           (0, max_transcript_len - x.transcript_len,
                                            0, max_boxes_num_batch - x.boxes_num),
                                           value=keys_vocab_cls.stoi['<pad>'])
                                     for i, x in enumerate(batch_list)]
        text_segments_batch_tensor = torch.stack(text_segments_padded_list, dim=0)

        
        text_length_padded_list = [F.pad(torch.LongTensor(x.text_segments[1]),
                                         (0, max_boxes_num_batch - x.boxes_num))
                                   for i, x in enumerate(batch_list)]
        text_length_batch_tensor = torch.stack(text_length_padded_list, dim=0)

        
        mask_padded_list = [F.pad(torch.ByteTensor(x.mask),
                                  (0, max_transcript_len - x.transcript_len,
                                   0, max_boxes_num_batch - x.boxes_num))
                            for i, x in enumerate(batch_list)]
        mask_batch_tensor = torch.stack(mask_padded_list, dim=0)

        if self.training:
            
            iob_tags_label_padded_list = [F.pad(torch.LongTensor(x.iob_tags_label),
                                                (0, max_transcript_len - x.transcript_len,
                                                 0, max_boxes_num_batch - x.boxes_num),
                                                value=iob_labels_vocab_cls.stoi['<pad>'])
                                          for i, x in enumerate(batch_list)]
            iob_tags_label_batch_tensor = torch.stack(iob_tags_label_padded_list, dim=0)

        else:
           
            image_indexs_list = [x.image_index for x in batch_list]
            image_indexs_tensor = torch.tensor(image_indexs_list)

        if self.training:
            
            batch = dict(whole_image=image_batch_tensor,
                         relation_features=relation_features_batch_tensor,
                         text_segments=text_segments_batch_tensor,
                         text_length=text_length_batch_tensor,
                         boxes_coordinate=boxes_coordinate_batch_tensor,
                         mask=mask_batch_tensor,
                         iob_tags_label=iob_tags_label_batch_tensor)
        else:
            
            batch = dict(whole_image=image_batch_tensor,
                         relation_features=relation_features_batch_tensor,
                         text_segments=text_segments_batch_tensor,
                         text_length=text_length_batch_tensor,
                         boxes_coordinate=boxes_coordinate_batch_tensor,
                         mask=mask_batch_tensor,
                         image_indexs=image_indexs_tensor)

        return batch
