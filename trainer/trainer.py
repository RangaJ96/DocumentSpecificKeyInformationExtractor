import os
import numpy as np
from numpy import inf

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import inf_loop
from utils.metrics import MetricTracker, SpanBasedF1MetricTracker
from logger import TensorboardWriter
from utils.class_utils import iob_labels_vocab_cls
from utils.util import iob_tags_to_union_iob_tags


class Trainer:

    def __init__(self, model, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, max_len_step=None):

        self.config = config
        self.local_master = config['local_rank'] == 0
        self.logger = config.get_logger(
            'trainer', config['trainer']['log_verbosity']) if self.local_master else None

        self.device, self.device_ids = self._prepare_device(
            config['local_rank'], config['local_world_size'])
        self.model = model.to(self.device)

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        monitor_open = cfg_trainer['monitor_open']
        if monitor_open:
            self.monitor = cfg_trainer.get('monitor', 'off')
        else:
            self.monitor = 'off'

        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            self.early_stop = inf if self.early_stop == -1 else self.early_stop

        self.start_epoch = 1

        if self.local_master:
            self.checkpoint_dir = config.save_dir

            self.writer = TensorboardWriter(
                config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        if self.config['trainer']['sync_batch_norm']:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)
        self.mode = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                        find_unused_parameters=True)

        self.data_loader = data_loader
        if max_len_step is None:

            self.len_step = len(self.data_loader)
        else:

            self.data_loader = inf_loop(data_loader)
            self.len_step = max_len_step
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        log_step = self.config['trainer']['log_step_interval']
        self.log_step = log_step if log_step != -1 and 0 < log_step < self.len_step else int(
            np.sqrt(data_loader.batch_size))

        val_step_interval = self.config['trainer']['val_step_interval']

        self.val_step_interval = val_step_interval

        self.gl_loss_lambda = self.config['trainer']['gl_loss_lambda']

        self.train_loss_metrics = MetricTracker('loss', 'gl_loss', 'crf_loss',
                                                writer=self.writer if self.local_master else None)
        self.valid_f1_metrics = SpanBasedF1MetricTracker(iob_labels_vocab_cls)

    def train(self):
       
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            self.data_loader.sampler.set_epoch(epoch)
            result_dict = self._train_epoch(epoch)

            if self.do_validation:
                val_result_dict = result_dict['val_result_dict']
                val_res = SpanBasedF1MetricTracker.dict2str(val_result_dict)
            else:
                val_res = ''

            best = False
            if self.monitor_mode != 'off' and self.do_validation:
                best, not_improved_count = self._is_best_monitor_metric(
                    best, not_improved_count, val_result_dict)
                if not_improved_count > self.early_stop:
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _is_best_monitor_metric(self, best, not_improved_count, val_result_dict):

        entity_name, metric = self.monitor_metric.split('-')
        val_monitor_metric_res = val_result_dict[entity_name][metric]
        try:

            improved = (self.monitor_mode == 'min' and val_monitor_metric_res <= self.monitor_best) or \
                       (self.monitor_mode ==
                        'max' and val_monitor_metric_res >= self.monitor_best)
        except KeyError:

            self.monitor_mode = 'off'
            improved = False
        if improved:
            self.monitor_best = val_monitor_metric_res
            not_improved_count = 0
            best = True
        else:
            not_improved_count += 1
        return best, not_improved_count

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_loss_metrics.reset()

        for step_idx, input_data_item in enumerate(self.data_loader):
            step_idx += 1
            for key, input_value in input_data_item.items():
                if input_value is not None:
                    input_data_item[key] = input_value.to(
                        self.device, non_blocking=True)
            if self.config['trainer']['anomaly_detection']:

                with torch.autograd.detect_anomaly():
                    self.optimizer.zero_grad()

                    output = self.model(**input_data_item)

                    gl_loss = output['gl_loss']
                    crf_loss = output['crf_loss']
                    total_loss = torch.sum(
                        crf_loss) + self.gl_loss_lambda * torch.sum(gl_loss)

                    total_loss.backward()

                    self.optimizer.step()
            else:
                self.optimizer.zero_grad()

                output = self.model(**input_data_item)

                gl_loss = output['gl_loss']
                crf_loss = output['crf_loss']
                total_loss = torch.sum(crf_loss) + \
                    self.gl_loss_lambda * torch.sum(gl_loss)

                total_loss.backward()

                self.optimizer.step()

            dist.barrier()

            dist.all_reduce(total_loss, op=dist.reduce_op.SUM)

            size = dist.get_world_size()
            gl_loss /= size
            crf_loss /= size

            avg_gl_loss = torch.mean(gl_loss)
            avg_crf_loss = torch.mean(crf_loss)
            avg_loss = avg_crf_loss + self.gl_loss_lambda * avg_gl_loss

            self.writer.set_step(
                (epoch - 1) * self.len_step + step_idx - 1) if self.local_master else None
            self.train_loss_metrics.update('loss', avg_loss.item())
            self.train_loss_metrics.update(
                'gl_loss', avg_gl_loss.item() * self.gl_loss_lambda)
            self.train_loss_metrics.update('crf_loss', avg_crf_loss.item())

            if self.do_validation and step_idx % self.val_step_interval == 0:
                val_result_dict = self._valid_epoch(epoch)

                best, not_improved_count = self._is_best_monitor_metric(
                    False, 0, val_result_dict)
                if best:
                    self._save_checkpoint(epoch, best)

            if step_idx == self.len_step + 1:
                break

        log = self.train_loss_metrics.result()

        if self.do_validation:
            val_result_dict = self._valid_epoch(epoch)
            log['val_result_dict'] = val_result_dict

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):

        self.model.eval()
        self.valid_f1_metrics.reset()
        with torch.no_grad():
            for step_idx, input_data_item in enumerate(self.valid_data_loader):
                for key, input_value in input_data_item.items():
                    if input_value is not None:
                        input_data_item[key] = input_value.to(
                            self.device, non_blocking=True)

                output = self.model(**input_data_item)
                logits = output['logits']
                new_mask = output['new_mask']
                if hasattr(self.model, 'module'):

                    best_paths = self.model.module.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                                  logits_batch_first=True)
                else:
                    best_paths = self.model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                           logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + step_idx, 'valid') \
                    if self.local_master else None

                predicted_tags_hard_prob = logits * 0
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        predicted_tags_hard_prob[i, j, tag_id] = 1

                golden_tags = input_data_item['iob_tags_label']
                mask = input_data_item['mask']
                union_iob_tags = iob_tags_to_union_iob_tags(golden_tags, mask)

                dist.barrier()  #
                self.valid_f1_metrics.update(
                    predicted_tags_hard_prob.long(), union_iob_tags, new_mask)

        f1_result_dict = self.valid_f1_metrics.result()

        self.model.train()

        return f1_result_dict

    def average_gradients(self, model):

        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def logger_info(self, msg):
        self.logger.info(msg) if self.local_master else None

    def logger_warning(self, msg):
        self.logger.warning(msg) if self.local_master else None

    def _prepare_device(self, local_rank, local_world_size):

        ngpu_per_process = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * ngpu_per_process,
                          (local_rank + 1) * ngpu_per_process))

        if torch.cuda.is_available() and local_rank != -1:
            torch.cuda.set_device(device_ids[0])
            device = 'cuda'
        else:
            device = 'cpu'
        device = torch.device(device)
        return device, device_ids

    def _save_checkpoint(self, epoch, save_best=False):

        if not self.local_master:
            return

        if hasattr(self.model, 'module'):
            arch = type(self.model.module).__name__
            state_dict = self.model.module.state_dict()
        else:
            arch = type(self.model).__name__
            state_dict = self.model.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)

        else:
            filename = str(self.checkpoint_dir /
                           'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):

        resume_path = str(resume_path)

        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        self.model.load_state_dict(checkpoint['state_dict'])
