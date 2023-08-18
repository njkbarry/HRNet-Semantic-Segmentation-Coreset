from transformers import ViTFeatureExtractor, ViTModel
import sys
import os
from pathlib import Path

# Add directory roots to path for this repo and cords submodule
# FIXME: Is there a better way to do this
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/cords")
sys.path.insert(0, os.getcwd() + "/lib")

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from lib.datasets import pascal_ctx
from analysis.rank_correlation.utils import (
    get_rank_corr_dataset,
    get_embeddings,
    get_sim_kernel,
    SubModularFunction,
)
from scipy.stats import spearmanr
import itertools
from collections import defaultdict
from typing import List
import argparse


def get_submod_rank(model, images, device, metric, submod_function, dataset, training):
    embeddings = get_embeddings(
        model=model, dataset=dataset, device=device, images=images
    )
    sim_kernel = get_sim_kernel(
        metric=metric, submod_function=submod_function, embeddings=embeddings
    )
    function = SubModularFunction(
        function_type=submod_function,
        n=embeddings.shape[0],
        sim_kernel=sim_kernel,
        dataset=dataset,
        data_subset="train" if training else "val",
        metric=metric,
        model=model,
    )
    rank = function.get_order()
    return rank, embeddings, sim_kernel


def submod_rank_corr_run(
    model_x, model_y, images, device, training, metric, submod_function, dataset
):
    """
    NOTE:
        - Currently experiment only varies model for feature space and keeps all other variables constant.
        - This may change but is computationally exponential in the number of variables
    """
    rank_x, emb_x, sim_kernel_x = get_submod_rank(
        model=model_x,
        images=images,
        device=device,
        metric=metric,
        submod_function=submod_function,
        dataset=dataset,
        training=training,
    )
    rank_y, emb_y, sim_kernel_y = get_submod_rank(
        model=model_y,
        images=images,
        device=device,
        metric=metric,
        submod_function=submod_function,
        dataset=dataset,
        training=training,
    )
    corr = spearmanr(rank_x, rank_y)
    # spearmanr(rank_x[:115], rank_y[:115]) seems to deliver a reasonable p-value
    return corr


if __name__ == "__main__":
    """
    Determine Spearman's Rank Correlation Coefficients for the orderings of each
    feature embedding model for a given submodular funciton and similarity kernel.
    Saves results as a .txt file
    """

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(",")

    parser = argparse.ArgumentParser(description="Rank Correlation Experiment")

    parser.add_argument(
        "--dataset",
        help="dataset name",
        default="pascal_ctx",
        type=str,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--train_set",
        help="whether to use the training set, else the validation",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Similarity metric to generate kernel",
        required=True,
    )
    parser.add_argument(
        "--submod_function",
        type=str,
        help="Submodular function to generate rankings",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=list_of_strings,
        help="Models used as feature embedders to run experiment over all combinations of.",
        required=True,
    )

    args = parser.parse_args()

    """
    # TOOD:
        - Replace with args
    """

    # dataset = "pascal_ctx"
    # device = "cpu"
    # train_set = True
    # metric = "cossim"
    # submod_function = "gc"

    # models = ["ViT", "oracle_spat"]
    # models = [
    #     "ViT",
    #     "ViT_cls",
    #     "oracle_spat",
    #     "oracle_context",
    #     "clip",
    #     "segformer",
    #     "sam",
    #     "dino",
    #     "dino_cls",
    # ]

    images, _ = get_rank_corr_dataset(dataset=args.dataset, training=args.train_set)

    experiment_results = defaultdict(list)

    for model_x, model_y in itertools.combinations(args.models, 2):
        corr, p_val = submod_rank_corr_run(
            model_x=model_x,
            model_y=model_y,
            images=images,
            device=args.device,
            training=args.train_set,
            metric=args.metric,
            submod_function=args.submod_function,
            dataset=args.dataset,
        )
        experiment_results["model_x"].append(model_x)
        experiment_results["model_y"].append(model_y)
        experiment_results["corr"].append(corr)
        experiment_results["p_val"].append(p_val)

    df = pd.DataFrame(data=experiment_results)
    experimental_results_path = (
        args.dataset + "training"
        if args.train_set is True
        else "val" + args.metric + args.submod_function + ".csv"
    )
    df.to_csv(experimental_results_path)
