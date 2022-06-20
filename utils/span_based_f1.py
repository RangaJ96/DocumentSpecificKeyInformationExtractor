from typing import *
from collections import defaultdict

import torch

from torchtext.vocab import Vocab

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics.metric import Metric

from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    bioul_tags_to_spans,
    iob1_tags_to_spans,
    bmes_tags_to_spans,
    TypedStringSpan
)



TAGS_TO_SPANS_FUNCTION_TYPE = Callable[
    [List[str], Optional[List[str]]], List[TypedStringSpan]]  


class SpanBasedF1Measure(Metric):
    

    def __init__(self,
                 vocab: Vocab = None,
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None) -> None:
        

        self._label_encoding = label_encoding
        self._tags_to_spans_function = tags_to_spans_function
        
        self._label_vocabulary = vocab
        self._ignore_classes: List[str] = ignore_classes or []

        
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        
        self._false_negatives: Dict[str, int] = defaultdict(int)
        
        self._total: Dict[str, int] = defaultdict(int)

        self.mapped_class = []
        for k, v in self._label_vocabulary.stoi.items():
            if k == '<pad>' or k == '<unk>':
                self.mapped_class.append(self._label_vocabulary.stoi['O'])
            else:
                self.mapped_class.append(v)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 prediction_map: Optional[torch.Tensor] = None):
       
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask, prediction_map = self.detach_tensors(predictions,
                                                                             gold_labels,
                                                                             mask, prediction_map)
        num_classes = predictions.size(-1)
        
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).long()
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is None:
            batch_size = gold_labels.size(0)
            prediction_map = torch.tensor([self.mapped_class for i in range(batch_size)]).long().to(gold_labels.device)

        argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
        gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
               
                continue

            predicted_string_labels = [self._label_vocabulary.itos[int(label_id)]
                                       for label_id in sequence_prediction[:length].tolist()]
            gold_string_labels = [self._label_vocabulary.itos[int(label_id)]
                                  for label_id in sequence_gold_label[:length].tolist()]

            

            tags_to_spans_function = None
           
            if self._label_encoding is None and self._tags_to_spans_function:
                tags_to_spans_function = self._tags_to_spans_function
            
            elif self._label_encoding == "BIO":
                tags_to_spans_function = bio_tags_to_spans
                
            elif self._label_encoding == "IOB1":
                tags_to_spans_function = iob1_tags_to_spans
            

            predicted_spans = tags_to_spans_function(predicted_string_labels, self._ignore_classes)
            
            gold_spans = tags_to_spans_function(gold_string_labels, self._ignore_classes)

            predicted_spans = self.continue_spans(predicted_spans)
            
            gold_spans = self.continue_spans(gold_spans)

            for span in gold_spans:
                self._total[span[0]] += 1

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
          
          
            for span in gold_spans:
                self._false_negatives[span[0]] += 1


    def continue_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
        
        span_set: Set[TypedStringSpan] = set(spans)
        continued_labels: List[str] = [label[2:] for (label, span) in span_set if label.startswith("C-")]
        
        for label in continued_labels:
            continued_spans = {span for span in span_set if label in span[0]}

            span_start = min(span[1][0] for span in continued_spans)
            span_end = max(span[1][1] for span in continued_spans)
            replacement_span: TypedStringSpan = (label, (span_start, span_end))

            span_set.difference_update(continued_spans)
            span_set.add(replacement_span)

        return list(span_set)

    def get_metric(self, reset: bool = False):
        
        tags: Set[str] = set()
        
        tags.update(self._true_positives.keys())
        
        tags.update(self._false_positives.keys())
        
        tags.update(self._false_negatives.keys())
        
        tags.update(self._total.keys())
        
        mat = {}
        
        for tag in tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "mEP" + "-" + tag
            recall_key = "mER" + "-" + tag
            f1_key = "mEF" + "-" + tag
            accuracy_key = "mEA" + "-" + tag
            mat[precision_key] = precision
            mat[recall_key] = recall
            mat[f1_key] = f1_measure

            mat[accuracy_key] = self._true_positives[tag] / (self._total[tag] + 1e-13)

        
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        mat["mEP-overall"] = precision
        mat["mER-overall"] = recall
        mat["mEF-overall"] = f1_measure

        if sum(self._total.values()) != 0:
            mat["mAE-overall"] = sum(self._true_positives.values()) / sum(self._total.values())
        else:
            mat["mAE-overall"] = 0

        if reset:
            self.reset()
        return mat

   
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        
        return precision, recall, f1_measure


    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
        self._total = defaultdict(int)
