# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------


import argparse
import os
import pprint
import shutil
import sys
import math

# Add directory roots to path for this repo and cords submodule
# FIXME: Is there a better way to do this
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/cords")

import logging
import time
import timeit
from pathlib import Path
from copy import deepcopy
import collections

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate, full_train_metric
from utils.modelsummary import get_model_summary
from utils.utils import (
    create_logger,
    FullModel,
    initialise_stochastic_subsets,
    initialise_global_order,
)

from cords.utils.data.dataloader.SL.adaptive import (
    AdaptiveRandomDataLoader,
    StochasticGreedyDataLoader,
    RandomDataLoader,
    WeightedRandomDataLoader,
    MILODataLoader,
)
from dotmap import DotMap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")

    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument("--seed", type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed

    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler

        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    if args.seed > 0:
        import random

        print("Seeding with", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        "writer": SummaryWriter(tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
        "fulltrain_global_steps": 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    print("args.local_rank: ", args.local_rank)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device("cuda:{}".format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    # build model
    model = eval("models." + config.MODEL.NAME + ".get_seg_model")(config)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # copy model file
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, "models")
        # if os.path.exists(models_dst_dir):
        #     shutil.rmtree(models_dst_dir)
        # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        train_batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
        test_batch_size = config.TEST.BATCH_SIZE_PER_GPU
    else:
        train_batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
        test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    train_dataset = eval("datasets." + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
        scale_factor=config.TRAIN.SCALE_FACTOR,
    )
    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    if config.TRAIN.CORESET_ALGORITHM is not None:
        full_trainloader = deepcopy(trainloader)

    extra_epoch_iters = 0
    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval("datasets." + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.EXTRA_TRAIN_SET,
            num_samples=None,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TRAIN.BASE_SIZE,
            crop_size=crop_size,
            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
            scale_factor=config.TRAIN.SCALE_FACTOR,
        )
        extra_train_sampler = get_sampler(extra_train_dataset)
        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=train_batch_size,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler,
        )
        extra_epoch_iters = np.int(
            extra_train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus)
        )

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval("datasets." + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1,
    )

    test_sampler = get_sampler(test_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler,
    )

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            thres=config.LOSS.OHEMTHRES,
            min_kept=config.LOSS.OHEMKEEP,
            weight=train_dataset.class_weights,
        )
    else:
        criterion = CrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights
        )

    model = FullModel(model, criterion)
    if distributed:
        # Multi-processing per gpu
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    else:
        # Multi-threading
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == "sgd":
        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [
                {"params": bb_lr, "lr": config.TRAIN.LR},
                {
                    "params": nbb_lr,
                    "lr": config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT,
                },
            ]
        else:
            params = [{"params": list(params_dict.values()), "lr": config.TRAIN.LR}]

        optimizer = torch.optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
        )
    else:
        raise ValueError("Only Support SGD optimizer")

    epoch_iters = int(
        train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus)
    )

    best_mIoU = 0
    last_epoch = 0
    previous_time = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, "checkpoint.pth.tar")
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={"cuda:0": "cpu"})
            previous_time = checkpoint["time"]
            best_mIoU = checkpoint["best_mIoU"]
            last_epoch = checkpoint["epoch"]
            dct = checkpoint["state_dict"]

            model.module.model.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("model.")
                }
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    if config.TRAIN.CORESET_ALGORITHM.lower() == "milo":
        """
        ############################## MILO Dataloader Additional Arguments ##############################
        """
        num_epochs = end_epoch - last_epoch

        dss_args = DotMap(
            dict(
                type="MILO",
                fraction=config.TRAIN.RANDOM_SUBSET,
                kw=0.1,
                gc_ratio=config.MILO.GC_RATIO,
                sge_submod_function=config.MILO.SGE_SUBMOD_FUNCTION,
                wre_submod_function=config.MILO.WRE_SUBMOD_FUNCTION,
                select_every=1,
                kappa=0,
                per_class=True,
                temperature=1,
                collate_fn=None,
                device=device,
                num_epochs=num_epochs,
                num_gpus=len(gpus),
                partition_mode=config.MILO.PARTITION_MODE,
                feature_embdedder=config.MILO.FEATURE_EMBEDDER,
                metric=config.MILO.METRIC,
            )
        )
        # subset_selection_name = (
        #     dss_args.type
        #     + "_"
        #     + dss_args.submod_function
        #     + "_"
        #     + str(dss_args.gc_ratio)
        #     + "_"
        #     + str(dss_args.kw)
        # )

        gc_stochastic_subsets_file_path = os.path.join(
            os.path.abspath("./data/preprocessing"),
            config["DATASET"]["DATASET"]
            + "_"
            + str(dss_args.feature_embdedder)
            + "_"
            + str(dss_args.metric)
            + "_"
            + str(dss_args.sge_submod_function)
            + "_"
            + str(dss_args.kw)
            + "_"
            + str(dss_args.fraction)
            + "_"
            + "stochastic_subsets.pkl",
        )
        global_order_file_path = os.path.join(
            os.path.abspath("./data/preprocessing"),
            config["DATASET"]["DATASET"]
            + "_"
            + str(dss_args.feature_embdedder)
            + "_"
            + str(dss_args.metric)
            + "_"
            + str(dss_args.wre_submod_function)
            + "_"
            + str(dss_args.kw)
            + "_global_order.pkl",
        )

        # FIXME: May be redundant
        # dss_args["subset_selection_name"] = subset_selection_name

        dss_args["global_order_file"] = global_order_file_path
        dss_args["gc_stochastic_subsets_file"] = gc_stochastic_subsets_file_path

        if not os.path.exists(gc_stochastic_subsets_file_path):
            initialise_stochastic_subsets(dss_args, config)

        #########################################################################
        # Development Code
        #########################################################################

        def dev_generate_seg_partitions(train_dataset):
            """
            The MILO algorithm implemented for classification task partitions the
            dataset based on image class, which is singular per image. In a
            semantic segmentation context that is not the case and so we must
            partition the dataset in some other way.
            https://trello.com/c/bDOosO5M/11-investigate-dataset-partitioning
            """
            df_path = "/home/njbarry/punim1896/coresets/repositories/HRNet-Semantic-Segmentation-Coreset/plots/label_counts_df.csv"
            if not os.path.exists(df_path):
                image_stats = {}
                for _, data in tqdm(
                    enumerate(train_dataset),
                    total=len(train_dataset),
                    desc="Gathering dataset pixel-wise label counts",
                ):
                    image, label, _, name = data
                    unique, counts = np.unique(label, return_counts=True)
                    image_stats[name] = dict(zip(unique, counts))

                Seg_Class_Counts = pd.DataFrame(image_stats)
                Seg_Class_Counts = Seg_Class_Counts.fillna(0)
                Seg_Class_Counts.to_csv(df_path)
            else:
                Seg_Class_Counts = pd.read_csv(df_path)

            clustering_model = KMeans(
                n_clusters=20  # Number of clusters arbitrarily chosen
            )
            # Fitting Model
            clustering_model.fit(Seg_Class_Counts.T)

            pca = PCA(n_components=3)
            X = pca.fit_transform(Seg_Class_Counts.T)
            fig = plt.figure(1, figsize=(8, 6))
            plt.clf()
            ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
            ax.set_position([0, 0, 0.95, 1])
            plt.cla()

            ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=clustering_model.labels_,
                cmap=plt.cm.nipy_spectral,
                edgecolor="k",
            )
            plot_dir = "/home/njbarry/punim1896/coresets/repositories/HRNet-Semantic-Segmentation-Coreset/plots/clustering_scatter"
            plt.savefig(plot_dir)

            # Assess clustering tendency
            from pyclustertend import ivat, hopkins
            from sklearn.preprocessing import scale

            X = scale(Seg_Class_Counts.T)
            hopkins(X, 150)
            ivat(X)

        # dev_generate_seg_partitions(train_dataset)

        #########################################################################

        if not os.path.exists(global_order_file_path):
            initialise_global_order(dss_args, config)

        trainloader = MILODataLoader(
            train_loader=trainloader,
            dss_args=dss_args,
            logger=logger,
            batch_size=train_batch_size,
            shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
            pin_memory=True,
            collate_fn=dss_args.collate_fn,
        )

        epoch_iters = int(np.floor(epoch_iters * config.TRAIN.RANDOM_SUBSET))

    elif config.TRAIN.CORESET_ALGORITHM.lower() == "adaptiverandom":
        """
        ############################## AdaptiveRandom Dataloader Additional Arguments ##############################
        """
        num_epochs = end_epoch - last_epoch

        # ala https://github.com/decile-team/cords/blob/main/configs/SL/config_adaptiverandom_mnist.py
        dss_args = DotMap(
            dict(
                type="AdaptiveRandom",
                fraction=config.TRAIN.RANDOM_SUBSET,
                select_every=1,
                kappa=0,
                collate_fn=None,
                device="cuda",
                num_epochs=num_epochs,
                num_gpus=len(gpus),
            )
        )

        trainloader = AdaptiveRandomDataLoader(
            train_loader=trainloader,
            dss_args=dss_args,
            logger=logger,
            batch_size=train_batch_size,
            shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
            pin_memory=True,
            collate_fn=dss_args.collate_fn,
        )

        epoch_iters = int(np.floor(epoch_iters * config.TRAIN.RANDOM_SUBSET))

    elif config.TRAIN.CORESET_ALGORITHM.lower() == "none":
        pass
    else:
        raise NotImplementedError

    for epoch in range(last_epoch, end_epoch):
        # exploitation_experiment = True
        # # Now using WRE, stop training
        # if config.TRAIN.CORESET_ALGORITHM.lower() == "milo" and exploitation_experiment and not trainloader.cur_epoch < math.ceil(trainloader.gc_ratio * trainloader.num_epochs):
        #     break
        # elif config.TRAIN.CORESET_ALGORITHM.lower() == "adaptiverandom" and exploitation_experiment and not epoch < math.ceil(config.MILO.GC_RATIO * end_epoch):
        #     break

        if epoch >= config.TRAIN.END_EPOCH:
            train(
                config,
                epoch - config.TRAIN.END_EPOCH,
                config.TRAIN.EXTRA_EPOCH,
                extra_epoch_iters,
                config.TRAIN.EXTRA_LR,
                extra_iters,
                extra_trainloader,
                optimizer,
                model,
                writer_dict,
            )
        else:
            train(
                config,
                epoch,
                config.TRAIN.END_EPOCH,
                epoch_iters,
                config.TRAIN.LR,
                num_iters,
                trainloader,
                optimizer,
                model,
                writer_dict,
            )

        # FIXME:
        # Remove dev code
        # torch.cuda.empty_cache()

        if (epoch + 1) % config.TRAIN.VAL_SAVE_EVERY == 0:
            valid_loss, mean_IoU, IoU_array = validate(
                config, testloader, model, writer_dict
            )

            if config.TRAIN.CORESET_ALGORITHM is not None:
                if args.local_rank <= 0:
                    logging.info(
                        "Warning: generting metrics on entire training set can significantly inflate training time"
                    )

                ft_valid_loss, ft_mean_IoU, ft_IoU_array = full_train_metric(
                    config, full_trainloader, model, writer_dict
                )

            if args.local_rank <= 0:
                logger.info(
                    "=> saving checkpoint to {}".format(
                        final_output_dir + "checkpoint.pth.tar"
                    )
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_mIoU": best_mIoU,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "time": (timeit.default_timer() - start) * len(gpus)
                        + previous_time,
                    },
                    os.path.join(final_output_dir, "checkpoint.pth.tar"),
                )
                if mean_IoU > best_mIoU:
                    best_mIoU = mean_IoU
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(final_output_dir, "best.pth"),
                    )
                msg = "Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}".format(
                    valid_loss, mean_IoU, best_mIoU
                )
                logging.info(msg)
                logging.info(IoU_array)

    if args.local_rank <= 0:
        torch.save(
            model.module.state_dict(), os.path.join(final_output_dir, "final_state.pth")
        )

        writer_dict["writer"].close()

    # Log wall-times of each gpu
    end = timeit.default_timer()
    total_time = (end - start) + previous_time
    logger.info(
        "GPU: {} - Hours: {}, Minutes: {}, Total seconds: {}".format(
            args.local_rank,
            int((end - start) / 3600),
            (int((end - start) / 60) - 60 * int((end - start) / 3600)),
            end - start,
        ),
    )

    if args.local_rank <= 0:
        logger.info("Done")


if __name__ == "__main__":
    main()
