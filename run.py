from augment.augment import *
from datasets.datasets import *
from engine.engine import *
from Imort_model.imort_model import *
from losses.losses import *
from samplers.samplers import *
from utils.utils import *


import argparse

import argparse

class Args:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 15000
        self.bce_loss = False
        self.unscale_lr = False
        # self.model = 'deit_base_patch16_224'
        self.model = 'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token'
        # self.model = 'vim_tiny_patch8_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token'

        self.input_size = 32
        self.drop = 0.0
        self.drop_path = 0.1
        self.model_ema = True
        self.model_ema_decay = 0.99996
        self.model_ema_force_cpu = False
        self.opt = 'adamw'
        self.opt_eps = 1e-8
        self.opt_betas = None
        self.clip_grad = None
        self.momentum = 0.9
        self.weight_decay = 0.05
        self.if_amp=True

        self.sched = 'cosine'
        self.lr = 5e-4
        self.lr_noise = None
        self.lr_noise_pct = 0.67
        self.lr_noise_std = 1.0
        self.warmup_lr = 1e-6
        self.min_lr = 1e-5

        self.decay_epochs = 30
        self.warmup_epochs = 5
        self.cooldown_epochs = 10
        self.patience_epochs = 10
        self.decay_rate = 0.1

        self.color_jitter = 0.3
        self.aa = 'rand-m9-mstd0.5-inc1'
        self.smoothing = 0.1
        self.train_interpolation = 'bicubic'

        self.repeated_aug = True
        self.train_mode = True
        self.ThreeAugment = True
        self.src = True

        self.reprob = 0.25
        self.remode = 'pixel'
        self.recount = 1
        self.resplit = False

        self.mixup = 0.8
        self.cutmix = 1.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'

        self.teacher_model = 'regnety_160'
        self.teacher_path = ''
        self.distillation_type = 'hard'
        self.distillation_alpha = 0.5
        self.distillation_tau = 1.0

        self.cosub = False

        self.finetune = ''
        self.attn_only = False

        # self.data_path = '/datasets01/imagenet_full_size/061417/'
        self.data_path = './data'
        self.data_set = 'CIFAR10'
        self.inat_category = 'name'

        self.output_dir = 'model'
        self.device = 'cuda'
        self.seed = 0
        self.resume = ''
        self.start_epoch = 0
        self.eval = False
        self.eval_crop_ratio = 0.875
        self.dist_eval = False
        self.num_workers = 1
        self.pin_mem = True

        self.distributed = False
        self.world_size = 1
        self.dist_url = 'env://'

        # self.if_amp = False
        self.local_rank = 0

        self.dataset = 'cifar10'
        self.num_labeled = 4000
        self.num_classes = 10
        self.resize = 32
        self.randaug = [2, 16]  # Example default values, change as needed
        self.expand_labels = True
        self.eval_step = 52
        self.workers = 1
        self.mu = 7
        self.nb_classes = 10
        self.threshold = 0.6
        self.lambda_u = 8
        self.eval_step = 300

# Example usage:
args = Args()
# print(args.batch_size)
# print(args.lr)



# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

# from datasets import build_dataset
# from engine import train_one_epoch, evaluate
# from losses import DistillationLoss
# from samplers import RASampler
# from augment import new_data_aug_generator

from contextlib import suppress

# import models_mamba

# import utils

# log about
# import mlflow




import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]








from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def main(args):
    # init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # log about
    run_name = args.output_dir.split("/")[-1]
    # if args.local_rank == 0 and args.gpu == 0:
    #     mlflow.start_run(run_name=run_name)
    #     for key, value in vars(args).items():
    #         mlflow.log_param(key, value)

    # dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # dataset_val, _ = build_dataset(is_train=False, args=args)


    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    # if args.ThreeAugment:
    #     data_loader_train.dataset.transform = new_data_aug_generator(args)

    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=sampler_val,
    #     batch_size=int(1.5 * args.batch_size),
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False
    # )



    labeled_dataset, unlabeled_dataset, test_dataset, finetune_dataset = DATASET_GETTERS[args.dataset](args)

    train_sampler = RandomSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

    data_loader_val = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)




    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')

    model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    loss_scaler_t = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        loss_scaler_t = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion_base = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion_base = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion_base = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion_base = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion_base = torch.nn.BCEWithLogitsLoss()


    teacher_model = None
    if args.distillation_type != 'none':
        # assert args.teacher_path, 'need to specify teacher-path when using distillation'
        # print(f"Creating teacher model: {args.teacher_model}")

        teacher_model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )
        optimizer_t = create_optimizer(args, teacher_model)
        lr_scheduler_t, _ = create_scheduler(args, optimizer_t)

        # teacher_model = create_model(
        #     args.teacher_model,
        #     pretrained=False,
        #     num_classes=args.nb_classes,
        #     global_pool='avg',
        # )
        # if args.teacher_path.startswith('https'):
        #     checkpoint = torch.hub.load_state_dict_from_url(
        #         args.teacher_path, map_location='cpu', check_hash=True)
        # else:
        #     checkpoint = torch.load(args.teacher_path, map_location='cpu')
        # teacher_model.load_state_dict(checkpoint['model'])

        teacher_model.to(device)
        # teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion_base, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint and args.if_amp: # change loss_scaler if not amp
                loss_scaler.load_state_dict(checkpoint['scaler'])
            elif 'scaler' in checkpoint and not args.if_amp:
                loss_scaler = 'none'
        lr_scheduler.step(args.start_epoch)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")
        return

    # log about
    # if args.local_rank == 0 and args.gpu == 0:
    #     mlflow.log_param("n_parameters", n_parameters)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)

        # return model
        # print(model)
        train_stats = train_one_epoch(
            model, teacher_model, criterion, criterion_base, labeled_loader, unlabeled_loader,
            optimizer, optimizer_t, device, epoch, loss_scaler, loss_scaler_t, amp_autocast,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        lr_scheduler_t.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_t': optimizer_t.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'lr_scheduler_t': lr_scheduler_t.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                    'scaler_t': loss_scaler_t.state_dict() if loss_scaler_t != 'none' else loss_scaler_t,
                    'args': args,
                }, checkpoint_path)

        if (epoch + 1) % args.eval_step == 0:
          test_stats = evaluate(data_loader_val, model, device, amp_autocast)
          print(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")

          if max_accuracy < test_stats["acc1"]:
              max_accuracy = test_stats["acc1"]
              if args.output_dir:
                  checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                  for checkpoint_path in checkpoint_paths:
                      save_on_master({
                          'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'epoch': epoch,
                          'model_ema': get_state_dict(model_ema),
                          'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                          'args': args,
                      }, checkpoint_path)

          print(f'Max accuracy: {max_accuracy:.2f}%')

          log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                      **{f'test_{k}': v for k, v in test_stats.items()},
                      'epoch': epoch,
                      'n_parameters': n_parameters}

        # log about
        # if args.local_rank == 0 and args.gpu == 0:
        #     for key, value in log_stats.items():
        #         mlflow.log_metric(key, value, log_stats['epoch'])


          if args.output_dir and is_main_process():
              with (output_dir / "log.txt").open("a") as f:
                  f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()

    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = 'cuda'
    args.gpu = 0
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    main(args)