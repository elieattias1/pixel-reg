# -*- coding: utf-8 -*-


import os, random, wandb, time, torch, models
import numpy as np
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from models import ResNet
from math import comb
import torch.nn.functional as F


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_taken = end - start
        hours = int(time_taken // 3600)
        minutes = int((time_taken % 3600) // 60)
        seconds = time_taken % 60
        print(
            f"Function {func.__name__!r} executed in {hours} hours, {minutes} minutes, {seconds:.2f} seconds"
        )
        return result

    return wrap_func


def name_wandb_run(task_config, reg_config):
    # set run name in wandb based on training id
    is_reg = reg_config["reg_alpha"] is not None
    task = task_config["task"]
    reg_data = reg_config["reg_data"]
    if reg_data == "none" and is_reg:
        reg_data = task
    reg_range = reg_config["reg_range"]
    type_of_reg = "boost" if reg_config["reg_boosting"] else "vanilla"
    return f"{task} - { f'{reg_range} - {type_of_reg} - reg - {reg_data}' if is_reg else 'noreg'} "


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ATanh(x, eps=1e-5):
    return torch.log(1 + x + eps) - torch.log(1 - x + eps)


# string for a duration
def time_str(t_elapse, progress=1.0):
    field_width = int(np.log10(t_elapse / 60 / progress)) + 1
    return "{{:{}d}}m{{:05.2f}}s".format(field_width).format(
        int(t_elapse // 60), t_elapse % 60
    )


# string for progress
def progress_str(i, total):
    field_width = int(np.log10(total)) + 1
    return "{{:{}d}}/{{:{}d}}".format(field_width, field_width).format(i, total)


def isotropic_downsampling(image):
    "image input shape : 144, 256"
    h = 4
    final = np.zeros((36, 64))
    for i in range(36):
        for j in range(64):
            final[i, j] = image[h * i, h * j]
    return final


def process_imagenet(
    image_id,
    ids,
    image_path="your_image_path",
):
    "Grayscale -> Cropping to 144x256 -> convolve by factor of 4 --> normalization"
    #   loading image
    image = Image.open(
        os.path.join(image_path, ids[image_id][:9], ids[image_id] + ".JPEG")
    )
    grayscale_image = torch.tensor(np.array(image.convert("L"))).float()

    #   cropping grayscaled to 144x256
    height, width = 144, 256
    center_y, center_x = grayscale_image.shape[0] // 2, grayscale_image.shape[1] // 2
    cropped = grayscale_image[
        center_y - height // 2 : center_y + height // 2,
        center_x - width // 2 : center_x + width // 2,
    ]

    #   Downsampling
    cropped_down = isotropic_downsampling(cropped)

    final_array = np.expand_dims(cropped_down, axis=0)

    return final_array


def get_images(ids):
    "fetching imagenet images to get pool.pt file"
    return np.array([process_imagenet(image_id, ids) for image_id in range(len(ids))])


def get_images_from_loader(loader):
    all_images = []
    all_labels = []
    for batch in loader:
        images, label = batch  # between 0 and 1
        for idx in range(len(images)):
            all_images.append(images[idx])
            all_labels.append(label[idx])

    all_images = torch.stack(all_images)  # between 0 and 1
    all_labels = torch.stack(all_labels)
    print(
        f"{len(all_images)} images, range : {[all_images.min().item(), all_images.max().item()]}"
    )
    return all_images, all_labels


def standardise_responses(responses):
    r_mean = responses.mean(axis=0)

    unit_responses = responses.clone()

    for i in range(responses.shape[0]):
        centered = unit_responses[i, :] - r_mean
        normalized = centered / torch.linalg.norm(centered)
        unit_responses[i, :] = normalized
    return unit_responses


def adjust_lr(epoch_num, optimizer, train_config, log: bool):
    r"""Adjusts learning rate

    Learning rates are adjusted in cycles, each cycle is divided into a number
    of phases, containing same number of epochs. within in each cycle, learning
    rate decays exponentially with respect to phases and be reset to initial
    value at the beginning of each cycle.
    This function is called at the beginning of each epoch.

    Args:
        epoch_num (int): number of epochs that were already trained
        optimizer (optimizer object): learning rate will be set
        train_config (dict): configuration containing descriptions of the schedule

    Returns:
        'lr' values for all param_groups of optimizer are updated
    """

    def lr_lambda(epoch_num, train_config):
        cycle_duration = train_config["phase_num"] * train_config["phase_duration"]
        phase_idx = (epoch_num % cycle_duration) // train_config["phase_duration"]
        return train_config["decay_rate"] ** phase_idx

    lr = train_config["lr"] * lr_lambda(epoch_num, train_config)
    print("learning rate {:.4g}".format(lr))
    if log:
        wandb.log({"learning_rate": lr})

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  # same learning rate for all parameter groups


# preprocessing of model inputs
def preprocess(
    dataset,
    batch,
    device,
    task,
):
    images, labels = batch

    if dataset in ["MNIST", "FashionMNIST"] and task in ["CIFAR10", "CIFAR100"]:
        images = (5.0 * (images - 0.5)).to(device)
    # if dataset in ["ImageNet"] and task in ["MNIST", "FashionMNIST"]:
    #     images = (images / 255.0).to(device)
    if dataset in ["ImageNet"] and task in ["CIFAR10", "CIFAR100"]:
        images = (5.0 * ((images / 255.0) - 0.5)).to(device)  # in [-2.5, 2.5] range

    if dataset in ["CIFAR10", "CIFAR100"]:
        images = (5.0 * (images - 0.5)).to(device)  # in [-2.5, 2.5] range

    # if dataset in ["ImageNet"] and task in ["CIFAR10", "CIFAR100"]:
    #     images = (5.0 * (images - 0.5)).to(device)

    return (images, labels)


def create_resnet_loaders(
    data_dir, task_config, train_config, num_workers=1, adversarial_augemntation=False
):

    if task_config["task"] == "MNIST":
        t_train = []
        t_test = []
    if task_config["task"] == "FashionMNIST":
        t_train = []
        t_test = []
    if task_config["task"].startswith("CIFAR"):
        t_train = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
        t_test = []
    if task_config["task"] == "ImageNet":
        t_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        t_test = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ]

    if not task_config["rgb"]:
        t_train.append(transforms.Grayscale())
        t_test.append(transforms.Grayscale())

    if adversarial_augemntation:
        t_train.append()

    t_train = transforms.Compose(t_train + [transforms.ToTensor()])
    t_test = transforms.Compose(t_test + [transforms.ToTensor()])

    if task_config["task"] == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(data_dir, download=True)
    if task_config["task"] == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(data_dir, download=True)
    if task_config["task"] == "ImageNet":
        dataset = torchvision.datasets.ImageNet(os.path.join(data_dir, "ImageNet2012"))
    if task_config["task"] == "MNIST":
        dataset = torchvision.datasets.MNIST(data_dir, download=True)
    if task_config["task"] == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST(data_dir, download=True)

    sample_num = len(dataset)
    valid_num = int(len(dataset) * train_config["valid_portion"])
    valid_idxs = random.sample(range(sample_num), valid_num)
    train_idxs = list(set(range(sample_num)) - set(valid_idxs))

    if task_config["task"] == "CIFAR10":
        dataset_train = torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(data_dir, train=True, transform=t_train),
            train_idxs,
        )
        dataset_valid = torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(data_dir, train=True, transform=t_test),
            valid_idxs,
        )
        dataset_test = torchvision.datasets.CIFAR10(
            data_dir, train=False, transform=t_test
        )
    if task_config["task"] == "CIFAR100":
        dataset_train = torch.utils.data.Subset(
            torchvision.datasets.CIFAR100(data_dir, train=True, transform=t_train),
            train_idxs,
        )
        dataset_valid = torch.utils.data.Subset(
            torchvision.datasets.CIFAR100(data_dir, train=True, transform=t_test),
            valid_idxs,
        )
        dataset_test = torchvision.datasets.CIFAR100(
            data_dir, train=False, transform=t_test
        )
    if task_config["task"] == "ImageNet":
        data_dir_image_net = "your_path_to_imagenet"
        dataset_train = torch.utils.data.Subset(
            torchvision.datasets.ImageNet(
                data_dir_image_net, split="train", transform=t_train
            ),
            train_idxs,
        )
        dataset_valid = torch.utils.data.Subset(
            torchvision.datasets.ImageNet(
                data_dir_image_net, split="train", transform=t_test
            ),
            valid_idxs,
        )
        dataset_test = torchvision.datasets.ImageNet(
            data_dir_image_net, split="val", transform=t_test
        )
    if task_config["task"] == "MNIST":
        dataset_train = torch.utils.data.Subset(
            torchvision.datasets.MNIST(data_dir, train=True, transform=t_train),
            train_idxs,
        )
        dataset_valid = torch.utils.data.Subset(
            torchvision.datasets.MNIST(data_dir, train=True, transform=t_test),
            valid_idxs,
        )
        dataset_test = torchvision.datasets.MNIST(
            data_dir, train=False, transform=t_test
        )
    if task_config["task"] == "FashionMNIST":
        dataset_train = torch.utils.data.Subset(
            torchvision.datasets.FashionMNIST(data_dir, train=True, transform=t_train),
            train_idxs,
        )
        dataset_valid = torch.utils.data.Subset(
            torchvision.datasets.FashionMNIST(data_dir, train=True, transform=t_test),
            valid_idxs,
        )
        dataset_test = torchvision.datasets.FashionMNIST(
            data_dir, train=False, transform=t_test
        )

    batch_size = train_config["resnet_batch_size"]
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return loader_train, loader_valid, loader_test


def model_state(resnet):
    return dict((name, param.cpu()) for name, param in resnet.state_dict().items())


def transformFactory(data_dir, task_config, reg_config, data):

    if data.startswith("CIFAR"):
        transform = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
    # if data == "ImageNet":
    #     transform = [
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #     ]
    if data == "ImageNet":
        transform = [
            transforms.CenterCrop((144, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((36, 64)),
        ]

    if data == "FashionMNIST":
        transform = []

    if data == "MNIST":
        transform = []

    if not task_config["rgb"]:
        transform.append(transforms.Grayscale())

    transform = transforms.Compose(transform + [transforms.ToTensor()])
    return transform


def dataset_regFactory(data_dir, reg_data, transform, reg_indices, reg_config):

    if reg_data == "CIFAR10":
        dataset_reg = torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform
            ),
            reg_indices,
        )

    if reg_data == "CIFAR100":
        dataset_reg = torch.utils.data.Subset(
            torchvision.datasets.CIFAR100(
                data_dir, train=True, download=True, transform=transform
            ),
            reg_indices,
        )

    if reg_data == "ImageNet":
        data_dir_image_net = "your_path_to_imagenet"
        dataset_reg = torch.utils.data.Subset(
            torchvision.datasets.ImageNet(
                data_dir_image_net, split="train", transform=transform
            ),
            reg_indices,
        )
        print("fetched reg_data")

    if reg_data == "MNIST":

        dataset_reg = torch.utils.data.Subset(
            torchvision.datasets.MNIST(
                data_dir, train=True, download=True, transform=transform
            ),
            reg_indices,
        )
    if reg_data == "FashionMNIST":

        dataset_reg = torch.utils.data.Subset(
            torchvision.datasets.FashionMNIST(
                data_dir, train=True, transform=transform, download=True
            ),
            reg_indices,
        )
    return dataset_reg


def fetch_reg_data(
    data_dir, task_config, reg_config, num_reg_images=5000, device="cpu"
):

    reg_data = reg_config["reg_data"]
    if not reg_data or reg_data == "none":
        print(
            f'reg_data not specified. Regularizing by default on {task_config["task"]}'
        )
        reg_config["reg_data"] = task_config["task"]
        reg_data = reg_config["reg_data"]
    print(f"regularizing on {reg_data}")
    transform = transformFactory(data_dir, task_config, reg_config, reg_data)

    if reg_data not in ["ImageNet"]:
        reg_indices = list(range(num_reg_images))
        dataset_reg = dataset_regFactory(
            data_dir, reg_data, transform, reg_indices, reg_config
        )
        shuffled_reg_indices = torch.randperm(num_reg_images).tolist()
        reg_subset = torch.utils.data.Subset(dataset_reg, shuffled_reg_indices)

        images = torch.stack([i[0] for i in reg_subset]).to(device)
        labels = torch.tensor([i[1] for i in reg_subset]).to(device)
    else:
        images, labels = dataset_regFactory(
            data_dir, reg_data, transform, None, reg_config
        )
        images = images[:num_reg_images].to(device)
    print(f"num_images : {images.shape}, range : {images.min()}, {images.max()}")

    return images, labels


def create_resnet_model(task_config, r_model_config):
    in_channels = 3 if task_config["rgb"] else 1

    if task_config["task"] in ["MNIST", "FashionMNIST"]:
        raw_size = 28
    if task_config["task"] in ["CIFAR10", "FashionMNIST", "MNIST"]:
        num_classes = 10

    if task_config["task"].startswith("CIFAR"):
        raw_size = 32
    if task_config["task"] == "ImageNet":
        raw_size = 224
    if task_config["task"] == "CIFAR100":
        num_classes = 100
    if task_config["task"] == "ImageNet":
        num_classes = 1000
    model_kwargs = {
        "in_shape": (in_channels, raw_size, raw_size),
        "num_classes": num_classes,
        "base_channels": r_model_config["base_channels"],
    }

    if r_model_config["archi"] == "ResNet18":
        resnet = models.ResNet18(**model_kwargs)
    if r_model_config["archi"] == "ResNet34":
        resnet = models.ResNet34(**model_kwargs)
    if r_model_config["archi"] == "ResNet50":
        resnet = models.ResNet50(**model_kwargs)
    if r_model_config["archi"] == "ResNet101":
        resnet = models.ResNet101(**model_kwargs)
    if r_model_config["archi"] == "ViT":
        resnet = models.ViT(**model_kwargs)
    if r_model_config["archi"] == "CVT7":
        resnet = models.CVT7(**model_kwargs)
    if r_model_config["archi"] == "CCT7":
        resnet = models.CCT7(**model_kwargs)
    return resnet


# pixel value of inputs are in [0, 1]
def noisy_inputs(inputs, noise_type, epsilon):
    new_inputs = inputs.clone()
    # min and max values for each image in the batch
    i_mins = [i.min() for i in new_inputs]
    i_maxs = [i.max() for i in new_inputs]
    assert min(i_mins) >= 0 and max(i_maxs) <= 1

    if noise_type not in ["Gaussian", "Uniform", "SaltPepper"]:
        raise ValueError("Noise type not supported")
    if noise_type == "Gaussian":
        new_inputs += torch.randn_like(new_inputs) * epsilon
    if noise_type == "Uniform":
        new_inputs += (torch.rand_like(new_inputs) - 0.5) * 2 * epsilon
    if noise_type == "SaltPepper":
        saltpepper = (torch.rand_like(new_inputs) < epsilon).to(torch.float) * (
            2 * torch.randint_like(new_inputs, 2) - 1
        )
        new_inputs += saltpepper

    i_clamped = torch.stack(
        [i.clamp(i_min, i_max) for i, i_min, i_max in zip(new_inputs, i_mins, i_maxs)]
    )

    return i_clamped


def categorical_s_matrix(all_labels, reg_data, val=1):
    if reg_data not in ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"]:
        raise ValueError(f"{reg_data} not supported for categorical regularization")

    labels = all_labels.clone()
    labels_expanded = labels.unsqueeze(1)  # Shape: [num_labels, 1]
    new_mat = labels_expanded == labels_expanded.T

    new_mat = new_mat.float()
    return new_mat * val


def boosting(s_mat, reg_config, val_1=1.0, val_2=0.0):

    threshold_1 = reg_config["reg_thresh"]
    threshold_2 = reg_config["reg_thresh_2"]
    new_s = torch.eye(*s_mat.shape)
    print(f"boosting regularization with threshold : {threshold_1}")
    if not threshold_2:
        new_s[s_mat > threshold_1] = val_1  # high similarity
        new_s[s_mat < -threshold_1] = -val_1
        return new_s

    elif threshold_2:
        # double thresholding is not working properly. TODO : fix this
        assert threshold_1 > threshold_2

        new_s[s_mat > threshold_1] = val_1
        new_s[s_mat < -threshold_1] = -val_1

        print((torch.abs(s_mat) < threshold_2).sum())
        new_s[torch.abs(s_mat) < threshold_2] = val_2

        return new_s


def select_indices(s_matrix, reg_config):
    threshold = reg_config["reg_thresh"]
    reg_range = reg_config["reg_range"]
    print(f"Selecting {reg_range} indices for regularization.")

    if reg_range not in [
        "all",
        "high",
        "low",
        "double",
        "positive",
        "negative",
        "high_positive",
    ]:
        raise ValueError(f"reg_range: {reg_range} not recognized")
    elif reg_range == "all":
        all_indices = torch.where(s_matrix > -100)
        indices_list = [
            (i, j)
            for i, j in zip(all_indices[0].tolist(), all_indices[1].tolist())
            if i < j
        ]

    elif reg_range == "high":
        print(f"threshold applied on similarity values: {threshold}")
        # high disimilarity
        high_similarity_indices = torch.where(s_matrix > threshold)

        # Get the indices where s_mat values are less than -threshold
        high_dissimilarity_indices = torch.where(s_matrix < -threshold)
        indices_list = [
            (i, j)
            for i, j in zip(
                high_similarity_indices[0].tolist(), high_similarity_indices[1].tolist()
            )
            if i < j
        ] + [
            (i, j)
            for i, j in zip(
                high_dissimilarity_indices[0].tolist(),
                high_dissimilarity_indices[1].tolist(),
            )
            if i < j
        ]

    elif reg_range == "low":
        low_similarity_indices = torch.where(torch.abs(s_matrix) < threshold)
        # Get the indices where s_mat values are in the range [-threshold, threshold]

        indices_list = [
            (i, j)
            for i, j in zip(
                low_similarity_indices[0].tolist(), low_similarity_indices[1].tolist()
            )
            if i < j
        ]
    elif reg_range == "double":
        low_similarity_indices = torch.where(
            torch.abs(s_matrix) < reg_config["reg_thresh_2"]
        )
        high_similarity_indices = torch.where(
            torch.abs(s_matrix) > reg_config["reg_thresh"]
        )

        print(low_similarity_indices[0].shape)
        print(high_similarity_indices[0].shape)

        indices_list = [
            (i, j)
            for i, j in zip(
                low_similarity_indices[0].tolist(), low_similarity_indices[1].tolist()
            )
            if i < j
        ] + [
            (i, j)
            for i, j in zip(
                high_similarity_indices[0].tolist(),
                high_similarity_indices[1].tolist(),
            )
            if i < j
        ]
    elif reg_range == "high_positive":
        high_similarity_indices = torch.where(s_matrix > threshold)
        indices_list = [
            (i, j)
            for i, j in zip(
                high_similarity_indices[0].tolist(), high_similarity_indices[1].tolist()
            )
            if i < j
        ]

    elif reg_range == "positive":
        high_similarity_indices = torch.where(s_matrix > threshold)
        low_similarity_indices = torch.where(torch.abs(s_matrix) < threshold)

        indices_list = [
            (i, j)
            for i, j in zip(
                high_similarity_indices[0].tolist(), high_similarity_indices[1].tolist()
            )
            if i < j
        ] + [
            (i, j)
            for i, j in zip(
                low_similarity_indices[0].tolist(),
                low_similarity_indices[1].tolist(),
            )
            if i < j
        ]

    elif reg_range == "negative":
        high_similarity_indices = torch.where(s_matrix < -threshold)
        low_similarity_indices = torch.where(torch.abs(s_matrix) < threshold)

        indices_list = [
            (i, j)
            for i, j in zip(
                high_similarity_indices[0].tolist(), high_similarity_indices[1].tolist()
            )
            if i < j
        ] + [
            (i, j)
            for i, j in zip(
                low_similarity_indices[0].tolist(),
                low_similarity_indices[1].tolist(),
            )
            if i < j
        ]

    print(
        "proportion of selected indices : ",
        round(
            100 * len(indices_list) / comb(s_matrix.shape[0], 2),
            4,
        ),
        "%",
    )
    return torch.tensor(indices_list)


def similarity_matrix(responses):
    def normalize(responses):
        r_scaled = responses - responses.mean(dim=0)
        r_unit = r_scaled / r_scaled.pow(2).sum(dim=1, keepdim=True).pow(0.5)
        return r_unit

    r_unit = normalize(responses)
    s_matrix = torch.matmul(r_unit, r_unit.t())
    return s_matrix


def load_model(save_dir: str, training_id: str, eval=True) -> ResNet:
    print(f"Model with training id : {training_id} is being loaded and attacked")
    info = torch.load(
        os.path.join(save_dir, "ckpts", "trainings_resnet", f"{training_id[:-2]}.pt")
    )[training_id]

    task_config = info["task_config"]
    reg_config = info["reg_config"]
    r_model_config = info["r_model_config"]

    resnet = create_resnet_model(task_config, r_model_config)
    _ = resnet.load_state_dict(info["best_state"])

    train_config = info["train_config"]
    if eval:
        resnet.eval()
    print(_)
    return resnet, train_config, task_config, reg_config


def get_s_matrix(reg_config, all_reg_images, labels, device):

    if all_reg_images.shape[1] == 3:
        print("RGB regularization images")
        all_reg_images = all_reg_images.mean(axis = 1)
        print(all_reg_images.shape)

    print("pixel representations of images")
    image_embeddings = torch.flatten(all_reg_images, start_dim=1).to(device)

    s_matrix = similarity_matrix(image_embeddings).to(device)

    print(f"similarity matrix built. Has shape : {s_matrix.shape}")

    indices_tensor = select_indices(
        s_matrix,
        reg_config=reg_config,
    )

    if reg_config["reg_boosting"]:

        s_matrix = boosting(
            s_matrix,
            reg_config=reg_config,
            val_1=reg_config["reg_boost_val_1"],
        )
    return s_matrix, indices_tensor


def get_reg_loader(reg_config, indices_tensor, all_reg_images):

    if reg_config["reg_data"] != "ImageNET":
        reg_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(indices_tensor),
            batch_size=reg_config["reg_batch_size"],
            shuffle=True,
            drop_last=True,
        )
        image_idxs = None
    else:

        image_idxs = torch.arange(reg_config["reg_image_num"])
        reg_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(all_reg_images, image_idxs),
            batch_size=reg_config["reg_batch_size"],
            shuffle=True,
            drop_last=True,
        )
    return reg_loader, image_idxs


def evaluate(loader, resnet, run_config, task_config):
    if run_config["device"] == "cuda:all":
        device = "cuda"
    else:
        device = run_config["device"]
    resnet.to(device)
    resnet.eval()

    criterion = torch.nn.CrossEntropyLoss()

    loss_total, count_total, count_correct = 0.0, 0, 0
    for batch in loader:
        inputs, labels = preprocess(
            task_config["task"], batch, device, task=task_config["task"]
        )
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)

        loss_total += loss.item() * inputs.shape[0]
        count_total += inputs.shape[0]
        count_correct += predicted.eq(labels).sum().item()
    loss = loss_total / count_total
    acc = 100.0 * count_correct / count_total
    print("cross entropy loss: {:.4f}, accuracy: {:6.2f}%".format(loss, acc))

    return loss, acc


def ATanhSquare(x, y, eps=1e-5):
    return (
        (
            torch.log(1 + x + eps)
            - torch.log(1 - x + eps)
            - torch.log(1 + y + eps)
            + torch.log(1 - y + eps)
        )
        .pow(2)
        .mean()
    )


def evaluate_all(
    resnet, run_config, task_config, loaders, names, losses, accs, wandb_log=False
):
    print("evaluating ...")
    for key, loader in zip(
        names,
        loaders,
    ):
        print(f"{key} dataset")

        epoch_loss, epoch_acc = evaluate(loader, resnet, run_config, task_config)
        if wandb_log:
            wandb.log(
                {
                    f"Cross Entropy Loss-{key}": epoch_loss,
                    f"acc-{key}": epoch_acc,
                    "Epoch": len(losses[key]),
                },
            )
        losses[key].append(epoch_loss)
        accs[key].append(epoch_acc)


def find_correctly_classified_in_batch(resnet, batch, device):
    indices_correctly_classified = []
    total = 0
    resnet = resnet.to(device)

    with torch.no_grad():
        X, Y = batch
        X, Y = X.to(device), Y.to(device)

        # Get model output
        output = resnet(X)
        # Get predictions from output
        final_pred = output.max(1)[1]  # Simplified without keepdim

        # Find indices of correctly classified examples
        correct = (final_pred == Y).nonzero(as_tuple=False).squeeze()
        # Gets the indices of correct predictions

        # Adjust indices for the batch offset
        # correct_indices = correct + i * X.size(0)  # Adjust indices for the batch

        # Store the correct indices
        indices_correctly_classified.extend(correct.tolist())

        # Update total count
        total += Y.size(0)
    clean_accuracy = len(indices_correctly_classified) / total
    print(f"Clean accuracy: {clean_accuracy}")
    return clean_accuracy, indices_correctly_classified


def find_correctly_classified(resnet, loader, task, device):
    indices_correctly_classified = []
    total = 0
    resnet = resnet.to(device)

    with torch.no_grad():
        for i, batch in enumerate(loader):  # Added enumeration to track the batch index
            X, Y = batch
            # Assuming utils.preprocess is defined and properly processing the input and labels
            X, Y = preprocess(
                dataset=task,
                batch=(X, Y),
                device=device,
                task=task,
            )
            X, Y = X.to(device), Y.to(device)

            # Get model output
            output = resnet(X)
            # Get predictions from output
            final_pred = output.max(1)[1]  # Simplified without keepdim

            # Find indices of correctly classified examples
            correct = (
                (final_pred == Y).nonzero(as_tuple=False).squeeze()
            )  # Gets the indices of correct predictions

            # Adjust indices for the batch offset
            correct_indices = correct + i * X.size(0)  # Adjust indices for the batch

            # Store the correct indices
            indices_correctly_classified.extend(correct_indices.tolist())

            # Update total count
            total += Y.size(0)

    # Calculate clean accuracy
    clean_accuracy = len(indices_correctly_classified) / total
    print(f"Clean accuracy: {clean_accuracy}")
    return clean_accuracy, indices_correctly_classified


attack_ranges_mnist = {
    "Gaussian": [0, 0.5],
    "Uniform": [0, 0.5],
    "SaltPepper": [0, 0.3],
    "TransferredFGSM": [0, 0.5],
    "BoundaryAttack": None,
}

attack_ranges_cifar10 = {
    "Gaussian": [0, 0.1],
    "Uniform": [0, 0.1],
    "SaltPepper": [0, 0.1],
    "TransferredFGSM": [0, 0.02],
    "BoundaryAttack": None,
}
attack_ranges_cifar100 = {
    "Gaussian": [0, 0.1],
    "Uniform": [0, 0.1],
    "SaltPepper": [0, 0.1],
    "TransferredFGSM": [0, 0.005],
    "BoundaryAttack": None,
}


def get_epsilon_range(task, attack):

    if task in ["MNIST", "FashionMNIST"]:
        return attack_ranges_mnist[attack]
    if task in ["CIFAR10"]:
        return attack_ranges_cifar10[attack]
    if task in ["CIFAR100"]:
        return attack_ranges_cifar100[attack]
