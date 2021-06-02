# -*- coding:utf-8 -*-
"""
Author:xufei
Date:2021/1/21
"""
import os, sys
import time
import torch
import shutil
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from DBnet.libs.tools import setup_logger
from DBnet.libs.network.utils import WarmupPolyLR, runningScore, cal_text_score
base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../"))
sys.path.append(base_dir)       # 设置项目根目录


class Train(object):
    def __init__(self,
                 parmas,
                 args,
                 model,
                 criterion,
                 train_loader,
                 validate_loader,
                 metric_cls=None,
                 post_process=None):
        super(Train, self).__init__()
        self.Scaler = GradScaler()  # (AMP)自动混合精度
        self.params = parmas

        # 设置模型存储路径
        save_output = os.path.abspath(os.path.join(base_dir, self.params.TRAIN.train_output_dir))
        save_name = "DBNet" + "_" + model.name
        self.save_dir = os.path.abspath(os.path.join(save_output, save_name))               # 训练模型保存路径
        self.checkpoint_dir = os.path.abspath(os.path.join(self.save_dir, 'checkpoints'))   # checkpoint保存路径
        print(self.checkpoint_dir)
        # 如何无重新训练或预训练参数
        if self.params.TRAIN.train_resume_checkpoint == '' and self.params.TRAIN.train_finetune_checkpoint == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # 初始化随机种子
        torch.manual_seed(self.params.TRAIN.random_seed)
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            self.device = torch.device('cuda:{}'.format(self.params.TRAIN.ID_GPU))
            np.random.seed(self.params.TRAIN.random_seed)
            torch.backends.cudnn.benchmark = True  # 保证每次方向传播参数结果一样
            torch.cuda.manual_seed(self.params.TRAIN.random_seed)
            torch.cuda.manual_seed_all(self.params.TRAIN.random_seed)
        else:
            self.with_cuda = False
            self.device = torch.device('cpu')

        # 打印训练模型及pytorch版本
        self.log_iter = self.params.TRAIN.train_log_iter  # log日志打印周期
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.logger = setup_logger(os.path.join(args.log_path, 'train.log'))
        self.logger_info('train with device {} and pytorch {}'.format(self.device, torch.__version__))

        # 训练集
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)
        self.epoch_result = None

        # 验证集
        if validate_loader is not None:
            assert metric_cls is not None
            assert post_process is not None
        self.validate_loader = validate_loader

        # 训练参数
        self.metric_cls = metric_cls
        self.post_process = post_process
        self.global_step = 0
        self.epochs = self.params.TRAIN.epochs
        self.start_epoch = 0

        # make inverse Normalize
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.UN_Normalize = True

        # 模型训练
        self.model = model
        self.model.to(self.device)
        self.criterion = criterion                      # 损失函数

        # 评价函数
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}

        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=parmas.TRAIN.opti_lr,
                                           weight_decay=parmas.TRAIN.opti_weight_decay,
                                           amsgrad=parmas.TRAIN.opti_amsgrad)
        # 学习率变化策略
        if self.params.TRAIN.lr_scheduler_type == 'WarmupPolyLR':
            warmup_iters = self.params.TRAIN.lr_scheduler_warmup_epoch * self.train_loader_len
            lr_dict = {}
            if self.start_epoch > 1:
                self.params.TRAIN.lr_scheduler_last_epoch = (self.start_epoch - 1) * self.train_loader_len
                lr_dict['last_epoch'] = self.params.TRAIN.lr_scheduler_last_epoch
            self.scheduler = WarmupPolyLR(optimizer=self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **lr_dict)

        # 重新训练或加载预训练
        if self.params.TRAIN.train_resume_checkpoint != '':
            self._load_checkpoints(os.path.join(self.checkpoint_dir, 'model_latest.pth'), resume=True)
        elif self.params.TRAIN.train_finetune_checkpoint != '':
            self._load_checkpoints(self.params.TRAIN.train_finetune_checkpoint, resume=False)

        # 统计加载数据
        if self.validate_loader is not None:
            s1 = f'train dataset has {len(self.train_loader.dataset)} samples,' \
                 f'{self.train_loader_len} in dataloader, ' \
                 f'validate dataset has {len(self.validate_loader.dataset)} samples,' \
                 f'{len(self.validate_loader)} in dataloader'
            self.logger_info(s1)
        else:
            self.logger_info(
                f"train dataset has {len(self.train_loader.dataset)} samples,{self.train_loader_len} in dataloader")

    def forward(self):
        for epoch in range(self.start_epoch+1, self.epochs+1):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.0
        running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']   # 动态调整学习率

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据转换并送入gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]
            with autocast():       # 使用半精度加速训练(autocast+GradScaler)
                preds = self.model(batch['img'])
                loss_dict = self.criterion(preds, batch)

            # backward
            self.optimizer.zero_grad()
            self.Scaler.scale(loss_dict['loss']).backward()
            self.Scaler.step(self.optimizer)
            self.Scaler.update()
            # loss_dict['loss'].backward()
            # self.optimizer.step()
            if self.params.TRAIN.lr_scheduler_type == 'WarmupPolyLR':
                self.scheduler.step()

            # acc, iou
            score_shrink_map = cal_text_score(preds[:, 0, :, :],
                                              batch['shrink_map'],
                                              batch['shrink_mask'],
                                              running_metric_text=running_metric_text,
                                              thred=self.params.TRAIN.post_processing_thresh)

            # loss and acc load log
            loss_str = 'loss: {:.4f}'.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            # 各个loss函数获取信息
            train_loss += loss_dict['loss']
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                s = '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_shrink_map: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                   epoch, self.epochs, i+1, self.train_loader_len, self.global_step, self.log_iter*cur_batch_size/batch_time, acc, iou_shrink_map, loss_str, lr, batch_time
                )
                self.logger_info(s)
                batch_start = time.time()
        return {'train_loss': train_loss/self.train_loader_len,
                'lr': lr,
                'time': time.time()-epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()

        # 释放GPU显存
        torch.cuda.empty_cache()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 将数据加载到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start_time = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start_time

                # 用QuadMetric对验证集进行评价
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        # 获取返回的评价参数:'precision','recall','fmeasure'
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame/total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        """
        训练+验证
        一个epoch训练完的log日志
        :return:None
        """
        finish_log = '[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'],
                                                                                self.epoch_result['time'], self.epoch_result['lr'])
        self.logger_info(finish_log)

        # 最后和最佳模型储存路径
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        best_net_save_path = '{}/model_best.pth'.format(self.checkpoint_dir)
        net_model_save_path = '{}/model_latest_model.pth'.format(self.checkpoint_dir)
        net_model_save_path_best = '{}/model_best_model.pth'.format(self.checkpoint_dir)

        self._save_checkpoints(self.epoch_result['epoch'], net_save_path)
        save_best = False
        # 存在验证集
        if self.validate_loader is not None and self.metric_cls is not  None:  # f1作为模型评价指标
            recall, precision, hmean = self._eval(self.epoch_result['epoch'])
            val_info = 'test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, hmean)
            self.logger_info(val_info)
            if hmean >= self.metrics['hmean']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['hmean'] = hmean
                self.metrics['precision'] = precision
                self.metrics['recall'] = recall
                self.metrics['best_model_epoch'] = self.epoch_result['epoch']
        # 不存在验证集
        else:
            if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model_epoch'] = self.epoch_result['epoch']
        best_str = 'current best'
        for k, v in self.metrics.items():
            best_str += f'{k}: {v}'
        self.logger_info(best_str)
        if save_best:
            import shutil
            shutil.copy(net_save_path, best_net_save_path)
            shutil.copy(net_model_save_path, net_model_save_path_best)
            self.logger_info("Saving current best: {}".format(best_net_save_path))
        else:
            self.logger_info("Saving checkpoint: {}".format(net_save_path))

    def inverse_normalize(self, batch_img):
        if self.UN_Normalize:
            batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * self.normalize_std[0] + self.normalize_mean[0]
            batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * self.normalize_std[1] + self.normalize_mean[1]
            batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * self.normalize_std[2] + self.normalize_mean[2]

    def _save_checkpoints(self, epoch, file_name):
        state_dict = self.model.state_dict()
        state = {'epoch': epoch,
                 'state_dict': state_dict,
                 'global_step': self.global_step,
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'config': self.params,
                 'metrics': self.metrics,
                 'models': self.model
                 }
        model_name = file_name.split('.')[0] + '_model.pth'
        filename = os.path.join(self.checkpoint_dir, file_name)
        model_name = os.path.join(self.checkpoint_dir, model_name)
        torch.save(state, filename)         # 保存模型及其它参数
        torch.save(self.model, model_name)  # 保存模型

    def _load_checkpoints(self, checkpoint_path, resume):
        self.logger_info(f'Loading checkpoint:{checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.params.TRAIN.lr_scheduler_type = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger_info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger_info("finetune from checkpoint {}".format(checkpoint_path))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info(f'{k}:{v}')
        self.logger_info('finish train')

    def logger_info(self, s):
        self.logger.info(s)
