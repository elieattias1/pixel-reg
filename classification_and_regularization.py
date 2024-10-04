# -*- coding: utf-8 -*-


import numpy as np
import torch, wandb, time
import utils
from utils import ATanhSquare


def train(
    resnet_loader,
    reg_loader,
    s_matrix,
    optimizer,
    resnet,
    all_reg_images,
    train_config,
    task_config,
    reg_config,
    run_config,
    wandb_log=False,
):

    device = run_config["device"]
    resnet.to(device)
    resnet.train()

    if reg_config["reg_alpha"]:
        reg_iterator = iter(reg_loader)
        s_matrix = s_matrix.to(device)

    cls_criterion = torch.nn.CrossEntropyLoss()
    if reg_config["arctanh"]:
        print("Using Arctanh^2 as regularization criterion")
        reg_criterion = ATanhSquare
    else:
        print("Using MSE as regularization criterion")
        reg_criterion = torch.nn.MSELoss()

    tic = time.time()
    batch_num, sample_num = len(resnet_loader), len(resnet_loader.dataset)
    for batch_id, batch in enumerate(resnet_loader, 1):

        resnet_images, labels = utils.preprocess(
            task_config["task"],
            batch,
            device,
            task=task_config["task"],
        )
        resnet_images, labels = resnet_images.to(device), labels.to(device)
        resnet_outputs = resnet(resnet_images)
        cls_loss = cls_criterion(resnet_outputs, labels)

        if reg_config["reg_alpha"]:
            reg_batch = next(reg_iterator, None)
            if reg_batch is None:
                reg_iterator = iter(reg_loader)
                reg_batch = next(reg_iterator, None)

            reg_indices = reg_batch[0]
            rows, cols = reg_indices[:, 0], reg_indices[:, 1]
            sim_targets = s_matrix[rows, cols]

            reg_images_row, _ = utils.preprocess(
                dataset=reg_config["reg_data"],
                batch=(all_reg_images[rows], None),
                device=device,
                task=task_config["task"],
            )

            reg_images_col, _ = utils.preprocess(
                dataset=reg_config["reg_data"],
                batch=(all_reg_images[cols], None),
                device=device,
                task=task_config["task"],
            )
            reg_images_row = reg_images_row.to(device)
            reg_images_col = reg_images_col.to(device)

            sim_outputs = resnet(reg_images_row, reg_images_col).to(device)

            reg_loss = reg_criterion(sim_outputs, sim_targets)
            loss = cls_loss + reg_config["reg_alpha"] * reg_loss
        else:
            loss = cls_loss
            reg_loss = torch.tensor(float("nan"))
        if wandb_log:
            wandb.log(
                {
                    "Batch-CrossEntropy Loss": cls_loss.item(),
                    "Batch-RegLoss": reg_loss.item(),
                    "Batch-Total Loss": loss.item(),
                }
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (
            batch_id % (-(-batch_num // run_config["disp_num"])) == 0
            or batch_id == batch_num
        ):
            _, predicted = resnet_outputs.max(1)
            acc = predicted.eq(labels).sum().item() / predicted.shape[0]
            if reg_config["reg_alpha"]:
                reg_corr = np.corrcoef(
                    sim_outputs.data.cpu().numpy(), sim_targets.cpu().numpy()
                )[0, 1]
            else:
                reg_corr = np.nan

            batch_str = "{{:{}d}} ({{:5.1f}}%):".format(
                int(np.log10(batch_num)) + 1
            ).format(
                batch_id,
                100.0 * min(1, batch_id * resnet_loader.batch_size / sample_num),
            )
            info_str = (
                batch_str
                + " [cls loss: {:.4f}] [acc:{:6.2f}%] [reg loss: {:.4f}] [corr: {:7.4f}] ({})".format(
                    cls_loss.item(),
                    100.0 * acc,
                    reg_loss.item(),
                    reg_corr,
                    utils.time_str(time.time() - tic),
                )
            )
            print(info_str)
