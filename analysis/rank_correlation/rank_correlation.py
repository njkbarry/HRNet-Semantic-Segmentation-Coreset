from transformers import ViTFeatureExtractor, ViTModel
from cords.cords.utils.data.data_utils.generate_global_order import (
    compute_image_embeddings, compute_vit_image_embeddings,
    compute_vit_cls_image_embeddings, compute_dino_image_embeddings, 
    compute_dino_cls_image_embeddings, get_rbf_kernel, get_cdist, get_dot_product
)
import submodlib
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
from PIL import Image
from lib.utils import DEFAULT_R2_COEFFICIENT


if __name__ == '__main__':
    """
    Determine rank of dataset
    """

    # Load dataset
    dataset = 'pascal_ctx'
    model = 'ViT'
    device = 'cpu'

    if dataset is 'pascal_ctx':
        train_dataset = eval("datasets." + dataset)(
            root='data/',
            list_path='train',
            num_samples=None,
            num_classes=60,
            multi_scale=True,
            flip=True,
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
        desc=f"loading {dataset} dataset for global ordering generation",
    ):
        train_images.append(
            Image.fromarray(np.transpose(x[0], (1, 2, 0)), mode="RGB")
        )
        train_labels.append(x[1])

    # Geenerate feature embeddings
    if model.lower() is 'oracle':
        # TODO: Add method to load oracle
        pass
    elif model.lower() is 'clip':
        # TODO: Unsure how to parameterise for image input
        # features = compute_image_embeddings()
        pass
    elif model.lower() is 'vit':
        embeddings = compute_vit_image_embeddings(images=train_images, device=device)
    elif model.lower() is 'dino':
        embeddings = compute_dino_image_embeddings(images=train_images, device=device)
    else:
        # TODO: Add other methods
        raise NotImplementedError
    
    # TODO: Add method to generate uncertainty measures
    metric = 'cossim'
    submod_function = 'gc'
    r2_coefficient = DEFAULT_R2_COEFFICIENT   # Define properly


    # Load submodular function
    if submod_function not in [
        "supfl",
        "gc_pc",
        "logdet_pc",
        "disp_min_pc",
        "disp_sum_pc",
    ]:
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

        data_knn = np.argsort(data_dist, axis=1)[:, :knn].tolist()
        data_r2 = np.nonzero(
            data_dist <= max(1e-5, data_dist.mean() - r2_coefficient * data_dist.std())
        )
        data_r2 = zip(data_r2[0].tolist(), data_r2[1].tolist())
        data_r2_dict = {}
        for x in data_r2:
            if x[0] in data_r2_dict.keys():
                data_r2_dict[x[0]].append(x[1])
            else:
                data_r2_dict[x[0]] = [x[1]]

    if submod_function == "fl":
        obj = submodlib.FacilityLocationFunction(
            n=embeddings.shape[0], separate_rep=False, mode="dense", sijs=data_sijs
        )

    elif submod_function == "logdet":
        obj = submodlib.LogDeterminantFunction(
            n=embeddings.shape[0], mode="dense", lambdaVal=1, sijs=data_sijs
        )

    elif submod_function == "gc":
        obj = submodlib.GraphCutFunction(
            n=embeddings.shape[0],
            mode="dense",
            lambdaVal=1,
            separate_rep=False,
            ggsijs=data_sijs,
        )

    elif submod_function == "disp_min":
        obj = submodlib.DisparityMinFunction(
            n=embeddings.shape[0], mode="dense", sijs=data_sijs
        )

    elif submod_function == "disp_sum":
        obj = submodlib.DisparitySumFunction(
            n=embeddings.shape[0], mode="dense", sijs=data_sijs
        )

    ranks = obj.get_order()
        