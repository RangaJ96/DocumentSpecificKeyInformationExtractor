import argparse
import torch
from tqdm import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model as arch_module
from data_utils.dataset import Dataset
from data_utils.dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']

    model = config.init_obj('model_arch', arch_module)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = Dataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None:
                    input_data_item[key] = input_value.to(device)

            output = model(**input_data_item)
            logits = output['logits']
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']
            text_segments = input_data_item['text_segments']
            mask = input_data_item['mask']

            best_paths = model.decoder.crf_layer.viterbi_tags(
                logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            decoded_tags_list = iob_index_to_str(predicted_tags)

            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):

                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])

                entities = []
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                result_file = output_path.joinpath(
                    Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(
                            item['entity_name'], item['text']))


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Testing')

    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')

    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')

    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')

    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')

    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')

    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')

    args = args.parse_args()

    main(args)
