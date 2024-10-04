# -*- coding: utf-8 -*-
import os, time
import utils
from numpy import r_
import torch.optim as optim

from dotenv import load_dotenv


from args_control import runtime_parser, get_configs
from records_management import TrainingEnvironment


from utils import (
    model_state,
    timer_func,
    name_wandb_run,
    get_s_matrix,
    get_reg_loader,
    evaluate_all,
    get_epsilon_range,
)
import wandb

from attack import attack_resnet
from classification_and_regularization import train


@timer_func
def main(
    data_dir,
    save_dir,
    run_config,
    task_config,
    r_model_config,
    train_config,
    reg_config,
    attack_config,
):
    device = run_config["device"]

    t_start = time.time()
    print(
        f"run_config = {run_config}\n\ntask_config = {task_config}\n\nr_model_config = {r_model_config}\n\ntrain_config = {train_config}\n\nreg_config = {reg_config}\n\nattack_config = {attack_config}"
    )
    training_environment = TrainingEnvironment(
        data_dir, save_dir, run_config, train_config
    )
    training_environment.setup()

    r_training_id = training_environment.records.fetch_id(
        {
            "task_config": task_config,
            "r_model_config": r_model_config,
            "train_config": train_config,
            "reg_config": reg_config,
            "attack_config": attack_config,
        }
    )
    print(f"training id: {r_training_id}")

    to_train = training_environment.to_train(run_config, r_training_id)
    if not to_train:
        return

    print(f"classifying : {task_config['task']}")

    if wandb_log:
        wandb.run.name = name_wandb_run(task_config, reg_config)

    attack_config["training_id"] = r_training_id

    utils.set_seed(train_config["seed"])

    # create dataloaders for classification pathway
    resnet_loader_train, resnet_loader_valid, resnet_loader_test = (
        utils.create_resnet_loaders(data_dir, task_config, train_config)
    )
    # create dataloaders for similarity pathway
    if not reg_config["reg_alpha"]:
        print("No regularization")
        s_matrix = None
        image_idxs = None
        reg_loader = None
        all_reg_images = None
    else:
        # prepare regularization
        all_reg_images, labels = utils.fetch_reg_data(
            data_dir,
            task_config,
            reg_config,
            num_reg_images=reg_config["reg_image_num"],
        )

        s_matrix, indices_tensor = get_s_matrix(
            reg_config, all_reg_images, labels, device=device
        )
        reg_loader, image_idxs = get_reg_loader(
            reg_config, indices_tensor, all_reg_images
        )

    # create resnet model
    resnet = utils.create_resnet_model(task_config, r_model_config).to(device)

    print("resnet model initialized")

    optimizer = optim.SGD(
        resnet.parameters(),
        lr=train_config['lr'],
        momentum=train_config["momentum"],
        weight_decay=train_config["weight_decay"],
    )

    epoch_num = 0

    losses = {"train": [], "valid": [], "test": []}
    accs = {"train": [], "valid": [], "test": []}

    evaluate_all(
        resnet,
        run_config,
        task_config,
        [resnet_loader_train, resnet_loader_valid, resnet_loader_test],
        ["train", "valid", "test"],
        losses,
        accs,
        wandb_log=wandb_log,
    )
    resnet_states = [model_state(resnet)]

    def save_progress():

        best_idx = losses["valid"].index(min(losses["valid"]))
        t_end = time.time()

        training_environment.metas.assign(
            r_training_id,
            {
                "task_config": task_config,
                "r_model_config": r_model_config,
                "train_config": train_config,
                "reg_config": reg_config,
                "attack_config": attack_config,
                "image_idxs": image_idxs,
                "epoch_num": epoch_num,
                "best_idx": best_idx,
                "losses": losses,
                "accs": accs,
                "t_start": t_start,
                "t_end": t_end,
                "finished": finished,
            },
        )
        training_environment.ckpts.assign(
            r_training_id,
            {
                "task_config": task_config,
                "r_model_config": r_model_config,
                "train_config": train_config,
                "best_state": resnet_states[best_idx],
                "reg_config": reg_config,
                "attack_config": attack_config,
            },
        )
        print("training progress saved")

    while True:
        finished = epoch_num == train_config["epoch_num"]
        if finished or epoch_num % run_config["save_period"] == 0:
            save_progress()

        if finished:
            print(f"training {r_training_id} finishes")
            break

        tic_epoch = time.time()
        print(f"\nepoch {epoch_num+1}")
        # adjust learning rate according to schedule
        utils.adjust_lr(epoch_num, optimizer, train_config, wandb_log)

        # train one epoch
        print("training ...")

        train(
            resnet_loader_train,
            reg_loader,
            s_matrix,
            optimizer,
            resnet,
            all_reg_images,
            train_config,
            task_config,
            reg_config,
            run_config,
            wandb_log=wandb_log,
        )

        epoch_num += 1
        evaluate_all(
            resnet,
            run_config,
            task_config,
            [resnet_loader_train, resnet_loader_valid, resnet_loader_test],
            ["train", "valid", "test"],
            losses,
            accs,
            wandb_log=wandb_log,
        )

        resnet_states.append(model_state(resnet))
        print(f"elapsed time {utils.time_str(time.time()-tic_epoch)}")

    attacks = [
        "Gaussian",
        "Uniform",
        "SaltPepper",
        # "TransferredFGSM", make sure an unregularized model is trained first
        "BoundaryAttack",
    ]
    resnet, train_config, task_config, reg_config = utils.load_model(
        save_dir, r_training_id
    )
    if attack_config["attack"]:

        results = {}
        for attack_type in attacks:
            print(f"\nPerforming {attack_type} attack")

            attack_config["attack_type"] = attack_type
            attack_config["epsilon_range"] = get_epsilon_range(
                task_config["task"], attack_type
            )

            result = attack_resnet(
                data_dir,
                save_dir,
                resnet,
                resnet_loader_test,
                task_config,
                reg_config,
                attack_config,
                wandb_log=wandb_log,
                device=device,
            )
            results[attack_type] = result
            training_environment.metas.update(
                r_training_id,
                {attack_type: result},
            )

    return r_training_id, results


wandb_log = False
wandb_project = "your_project_name"
reinit = True

if __name__ == "__main__":
    parser = runtime_parser("train_resnet")

    config = get_configs("train_resnet", parser.parse_args())

    load_dotenv()

    if wandb_log:
        # add wandb key to environment variables
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project=wandb_project,
            reinit=reinit,
            # track hyperparameters and run metadata
            config={
                **config[2],
                **config[3],
                **config[4],
                **config[5],
                **config[6],
                **config[7],
            },
        )
        wandb.define_metric("Perturbation_Strength")
        wandb.define_metric("Epoch")

    r_training_id, accuracies = main(*config)
    print(f"training id: {r_training_id}")
    print(f"accuracies: {accuracies}")
    if wandb_log:
        wandb.finish()
# #
