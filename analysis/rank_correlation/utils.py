import numpy as np
from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
from PIL import Image
from lib.datasets import pascal_ctx
import os
import submodlib
import pickle
from cords.utils.data.data_utils.generate_global_order import (
    compute_image_embeddings,
    compute_vit_image_embeddings,
    compute_vit_cls_image_embeddings,
    compute_dino_image_embeddings,
    compute_dino_cls_image_embeddings,
    get_rbf_kernel,
    get_cdist,
    get_dot_product,
    compute_oracle_image_embeddings,
    compute_segformer_image_embeddings,
    load_embeddings,
    store_embeddings,
)
from lib.utils.utils import (
    DEFAULT_R2_COEFFICIENT,
    DEFAULT_KNN_COEFFICIENT,
    DEFAULT_KW_COEFFICIENT,
)

EMBEDDINGS_PATH = "data/preprocessing/"


def get_rank_corr_dataset(dataset: str, training: bool = True):
    """
    Load dataset and process submodular funciton
    """
    if training:
        subset_path = "train"
    else:
        subset_path = "val"

    if dataset == "pascal_ctx":
        train_dataset = pascal_ctx(
            root="data/",
            list_path=subset_path,
            num_samples=None,
            num_classes=60,
            multi_scale=training,
            flip=training,
            ignore_label=-1,
            base_size=520,
            crop_size=(520, 520),
            downsample_rate=1,
            scale_factor=16,
        )
    else:
        raise NotImplementedError

    train_images = []
    train_labels = []
    for x in tqdm(
        train_dataset,
        total=len(train_dataset),
        desc=f"loading {dataset} dataset for rank correlation analysis",
    ):
        train_images.append(Image.fromarray(np.transpose(x[0], (1, 2, 0)), mode="RGB"))
        train_labels.append(x[1])

    return train_images, train_labels


def get_embeddings(model: str, dataset: str, device: str, images):
    # Generate feature embeddings
    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(EMBEDDINGS_PATH),
            dataset + "_" + model + "_train_embeddings.pkl",
        )
    ):
        if model == "ViT":
            train_embeddings = compute_vit_image_embeddings(images, device)
        elif model == "ViT_cls":
            train_embeddings = compute_vit_cls_image_embeddings(images, device)
        elif model == "dino":
            train_embeddings = compute_dino_image_embeddings(images, device)
        elif model == "dino_cls":
            train_embeddings = compute_dino_cls_image_embeddings(images, device)
        elif model == "oracle_spat":
            train_embeddings = compute_oracle_image_embeddings(
                images, device, mode="oracle_spat"
            )
        elif model == "oracle_context":
            train_embeddings = compute_oracle_image_embeddings(
                images, device, mode="oracle_context"
            )
        elif model == "sam":
            raise NotImplementedError
        elif model == "segformer":
            train_embeddings = compute_segformer_image_embeddings(images, device)
        elif model == "clip":
            train_embeddings = compute_image_embeddings(model, images, device)
        else:
            raise NotImplementedError
        store_embeddings(
            os.path.join(
                os.path.abspath(EMBEDDINGS_PATH),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(EMBEDDINGS_PATH),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    return train_embeddings


def get_sim_kernel(
    metric: str,
    embeddings: np.array,
    submod_function: str,
    kw: float = DEFAULT_KW_COEFFICIENT,
    knn: int = DEFAULT_KNN_COEFFICIENT,
    r2_coefficient: float = DEFAULT_R2_COEFFICIENT,
):
    """
    NOTE:
      - Currently limited to the submodular funcitons in SubModularFunction,
          due to the similarity kernel generation. In cords library following
          code defines the control flow logic.
      - Can delete this note after dev cycle

    if submod_function not in [
        "supfl",
        "gc_pc",
        "logdet_pc",
        "disp_min_pc",
        "disp_sum_pc",
    ]:
    """

    data_dist = get_cdist(embeddings)
    if metric == "rbf_kernel":
        data_sijs = get_rbf_kernel(data_dist, kw)
    elif metric == "dot":
        data_sijs = get_dot_product(embeddings)
        if submod_function in ["disp_min", "disp_sum"]:
            data_sijs = (data_sijs - np.min(data_sijs)) / (
                np.max(data_sijs) - np.min(data_sijs)
            )
        else:
            if np.min(data_sijs) < 0:
                data_sijs = data_sijs - np.min(data_sijs)
    elif metric == "cossim":
        normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        data_sijs = get_dot_product(normalized_embeddings)
        if submod_function in ["disp_min", "disp_sum"]:
            data_sijs = (data_sijs - np.min(data_sijs)) / (
                np.max(data_sijs) - np.min(data_sijs)
            )
        else:
            data_sijs = (data_sijs + 1) / 2
    else:
        raise ValueError("Please enter a valid metric")

    # data_knn = np.argsort(data_dist, axis=1)[:, :knn].tolist()
    # data_r2 = np.nonzero(
    #     data_dist <= max(1e-5, data_dist.mean() - r2_coefficient * data_dist.std())
    # )
    # data_r2 = zip(data_r2[0].tolist(), data_r2[1].tolist())
    # data_r2_dict = {}
    # for x in data_r2:
    #     if x[0] in data_r2_dict.keys():
    #         data_r2_dict[x[0]].append(x[1])
    #     else:
    #         data_r2_dict[x[0]] = [x[1]]

    return data_sijs


class SubModularFunction(ABC):
    def __init__(
        self,
        function_type: str,
        n: int,
        sim_kernel: np.ndarray,
        dataset: str,
        metric: str,
        data_subset: str,
        mode: str = "dense",
        pre_processing_dir: str = EMBEDDINGS_PATH,
    ) -> None:
        self.function_type = function_type
        self.n = n
        self.mode = mode
        self.sim_kernel = sim_kernel
        self.pre_processing_dir = pre_processing_dir
        self.dataset = dataset
        self.data_subset = data_subset
        self.metric = metric

        self.rankings_path = os.path.join(
            os.path.abspath(self.pre_processing_dir),
            self.dataset
            + "_"
            + self.function_type
            + self.metric
            + self.data_subset
            + "_rankings.pkl",
        )

        if self.function_type == "fl":
            self.function = submodlib.FacilityLocationFunction(
                n=self.n, separate_rep=False, mode=self.mode, sijs=self.sim_kernel
            )

        elif self.function_type == "logdet":
            self.function = submodlib.LogDeterminantFunction(
                n=self.n, mode=self.mode, lambdaVal=1, sijs=self.sim_kernel
            )

        elif self.function_type == "gc":
            self.function = submodlib.GraphCutFunction(
                n=self.n,
                mode=self.mode,
                lambdaVal=1,
                separate_rep=False,
                ggsijs=self.sim_kernel,
            )

        elif self.function_type == "disp_min":
            self.function = submodlib.DisparityMinFunction(
                n=self.n, mode=self.mode, sijs=self.sim_kernel
            )

        elif self.function_type == "disp_sum":
            self.function = submodlib.DisparitySumFunction(
                n=self.n, mode="dense", sijs=self.sim_kernel
            )

    @staticmethod
    def store_rankings(rankings_path: str, rankings: List[int]):
        """
        Write rankings to disc
        """
        with open(rankings_path, "wb") as fOut:
            pickle.dump({"rankings": rankings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_rankings(rankings_path: str):
        """
        Load rankings from disc
        """
        with open(rankings_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_rankings = stored_data["rankings"]
        return stored_rankings

    def get_order(self, embeddings: np.array) -> List[int]:
        if not os.path.exists(self.rankings_path):
            # Calculate rankings
            max_set = self.function.maximize(
                budget=embeddings.shape[0] - 1,
                optimizer="NaiveGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False,
            )

            order = [x[0] for x in max_set]

            self.store_rankings(self.rankings_path, order)
        else:
            # Load the rankings from disc
            train_embeddings = self.load_rankings(self.rankings_path)

        return order
