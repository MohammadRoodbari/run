# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

# from losses import DistillationLoss
# import utils

def train_one_epoch(model: torch.nn.Module, teacher_model: torch.nn.Module, criterion: DistillationLoss,base_criterion,
                    labeled_loader: Iterable, unlabeled_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_t: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_scaler_t, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    # model.train(set_training_mode)
    model.train()
    teacher_model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    # debug
    # count = 0
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for batch_idx in range(args.eval_step):


        labeled_iter = iter(labeled_loader)
        images_l, targets = next(labeled_iter)

        unlabeled_iter = iter(unlabeled_loader)
        (images_uw, images_us), _ = next(unlabeled_iter)


        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        images_l = images_l.to(args.device)
        targets = targets.to(args.device)

        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(images_l, targets)

        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with amp_autocast():
            # # print(samples.size())
            # print(f'images_us size : {images_us.size()}')
            # print(f'samples size : {samples.size()}')
            # print(f'label size : {images_l.size()}')
            # print(f'targets size : {targets.size()}')
            batch_size = images_l.shape[0]
            s_images = torch.cat((samples, images_us))
            outputs = model(s_images)

            t_images = torch.cat((samples, images_uw, images_us))
            outputs_t = teacher_model(t_images)

            t_logits_l = outputs_t[:batch_size]
            t_logits_uw, t_logits_us = outputs_t[batch_size:].chunk(2)
            del outputs_t

            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.distillation_tau, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )

            if not args.cosub:
                loss , t_loss_l = criterion(samples, hard_pseudo_label, t_logits_l, outputs, targets)


            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())

            s_loss_l_old = base_criterion(outputs[:batch_size].detach(), targets)

            t_loss_uda = t_loss_l + args.lambda_u * t_loss_u

        loss_value = loss.item()
        if loss_scaler != 'none':
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        with amp_autocast():
            with torch.no_grad():
                s_logits_l = model(images_l)
            # s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss_l_new = base_criterion(s_logits_l.detach(), targets)

            dot_product = s_loss_l_new - s_loss_l_old

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)

            t_loss = t_loss_uda + t_loss_mpl
            # t_loss = t_loss_mpl


        loss_value_t = t_loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        if loss_scaler != 'none':
            # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            is_second_order_t = hasattr(optimizer_t, 'is_second_order') and optimizer_t.is_second_order_t
            # loss_scaler(loss, optimizer, clip_grad=max_norm,
            #         parameters=model.parameters(), create_graph=is_second_order)

            loss_scaler_t(t_loss, optimizer_t, clip_grad=max_norm,
                    parameters=teacher_model.parameters(), create_graph=is_second_order_t)

        optimizer.zero_grad()
        optimizer_t.zero_grad()

        # else:
            # loss.backward()
            # t_loss_uda.backward()
            # if max_norm != None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            #     torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm)
            # optimizer.step()
            # optimizer_t.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_t=loss_value_t)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print(f'Train Epoch: {epoch}. batch epoch: {batch_idx}/{args.eval_step}. Loss: {loss_value:.4f}. Loss_t: {loss_value_t:.4f}')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
