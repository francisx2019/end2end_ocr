# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/27
"""
import time
import torch
import os
from pathlib import Path
import torch.nn.functional as F
from CRNN_CTC.libs.dataset.tools import get_batch_label


# ------------------------ set up logger 构建训练输出文件 ----------------------- #
def create_log_folder(cfg, phase='train'):
    root_output_dir = Path(cfg.TRAIN.log_path)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    checkpoints_output_dir = os.path.join(root_output_dir, 'checkpoint')

    tensorboard_log_dir = os.path.join(root_output_dir, 'log')

    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


# --------------------- 设置log日志打印 --------------------- #
def setup_logger(log_file_path: str = None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('CRNN_CTC.pytorch')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


# -------------- Plots a line-by-line description of a PyTorch model --------- #
def model_info(model):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    img, idx = train_loader.next()
    i = 0
    # for i, (img, idx) in enumerate(train_loader):
    while img is not None:
        # measure data time
        i += 1
        data_time.update(time.time() - end)

        labels = get_batch_label(dataset, idx)
        img = img.to(device)

        # inference
        preds = model(img)

        # compute loss
        batch_size = img.size(0)
        text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img.size(0))

        batch_time.update(time.time()-end)
        if i % config.TRAIN.print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=img.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)
            # setup_logger(os.path.join(config.TRAIN.log_path, 'train.log')).info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
        img. idx = train_loader.next()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    with torch.no_grad():
        # for i, (img, idx) in enumerate(val_loader):
        img, idx = val_loader.next()
        i = 0
        while img is not None:
            i += 1
            labels = get_batch_label(dataset, idx)
            img = img.to(device)

            # inference
            preds = model(img)

            # compute loss
            batch_size = img.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), img.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

            if (i + 1) % config.TRAIN.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.VAL.NUM_TEST_BATCH:
                break
            img, idx = val_loader.next()
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.VAL.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = config.VAL.NUM_TEST_BATCH * config.VAL.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)
    accuracy = n_correct / float(num_test_sample)

    msg = "[#correct:{} / #total:{}]， Test loss: {:.4f}, accuray: {:.4f}".format(n_correct, num_test_sample, losses.avg, accuracy)
    setup_logger(os.path.join(config.TRAIN.log_path, 'train.log')).info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer,
                 target_lr=0,
                 max_iters=0,
                 power=0.9,
                 warmup_factor=1./3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        # super(WarmupPolyLR, self).__init__()
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]
