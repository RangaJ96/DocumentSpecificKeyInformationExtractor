from collections import Counter
from torchtext.vocab import Vocab
from pathlib import Path

from . import entities_list


class ClassVocab(Vocab):

    def __init__(self, classes, specials=['<pad>', '<unk>'], **kwargs):
        
        
        cls_list = None
        
        if isinstance(classes, str):
            
            cls_list = list(classes)
            
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            
            with p.open(encoding='utf8') as f:
                classes = f.read()
                
                classes = classes.strip()
                
                cls_list = list(classes)
                
        elif isinstance(classes, list):
            cls_list = classes
            
        c = Counter(cls_list)
        
        self.special_count = len(specials)
        
        super().__init__(c, specials=specials, **kwargs)


def entities2iob_labels(entities: list):
    
    tags = []
    for e in entities:
        tags.append('B-{}'.format(e))
        tags.append('I-{}'.format(e))
        
    tags.append('O')
    
    return tags


keys_vocab_cls = ClassVocab(Path(__file__).parent.joinpath('key_set.txt'), specials_first=False)

iob_labels_vocab_cls = ClassVocab(entities2iob_labels(entities_list.Entities_list), specials_first=False)

entities_vocab_cls = ClassVocab(entities_list.Entities_list, specials_first=False)
