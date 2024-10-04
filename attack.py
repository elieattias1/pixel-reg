from adversarialAttacks.attackFactory import AttackFactory

import wandb, utils

# from dotenv import load_dotenv
from args_control import runtime_parser, get_configs
from utils import get_images_from_loader, timer_func, name_wandb_run
import numpy as np


def load_test_data(data_dir, task_config, train_config):
    _, _, loader_test = utils.create_resnet_loaders(
        data_dir, task_config, train_config, num_workers=1
    )
    X_test, Y_test = get_images_from_loader(loader_test)
    return X_test, Y_test


def get_training_id_from_path(path):
    with open(path) as f:
        u = f.readlines()
        for i in u:
            print(i)
            if i.startswith("training id"):
                training_id = i[12:].strip()
                return training_id
    return


@timer_func
def attack_resnet_from_path(
    data_dir, save_dir, path_to_training_prints, attack_config={}, wandb_log=False
):

    training_id = get_training_id_from_path(path_to_training_prints)
    if not training_id:
        raise ValueError("No training id found in path")

    resnet, train_config, task_config, reg_config = utils.load_model(
        save_dir=save_dir, training_id=training_id
    )

    if wandb_log:
        # wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.login(key="8f078b070b04444ee9cd9d7f8c19f2d166a91aa4")
        wandb.init(
            # set the wandb project where this run will be logged
            project="Adversarial Attacks",
            reinit=True,
            # track hyperparameters and run metadata
            config={**train_config, **task_config, **reg_config, **attack_config},
        )
        wandb.define_metric("Perturbation_Strength")
        # define which metrics will be plotted against it
        wandb.define_metric("Accuracy", step_metric="Perturbation_Strength")

    _, _, resnet_loader_test = utils.create_resnet_loaders(
        data_dir, task_config, train_config
    )

    X_test, Y_test = get_images_from_loader(resnet_loader_test)

    accuracies = attack_resnet(
        data_dir,
        save_dir,
        resnet,
        X_test,
        Y_test,
        task_config,
        reg_config,
        attack_config,
        wandb_log=wandb_log,
    )
    return accuracies


@timer_func
def attack_resnet(
    data_dir,
    save_dir,
    resnet,
    resnet_loader,
    task_config,
    reg_config,
    attack_config,
    wandb_log=False,
    device="cpu",
):
    resnet.eval()
    resnet.to(device)

    if wandb_log:
        wandb.run.name = name_wandb_run(task_config, reg_config)

    # initialize attack
    attack = AttackFactory(
        attack_config["attack_type"],
        task=task_config["task"],
        wandb_log=wandb_log,
        data_dir=data_dir,
        save_dir=save_dir,
        device=device,
    )

    # initialize epsilons
    if attack_config["attack_type"] != "BoundaryAttack":

        epsilons = np.linspace(
            *attack_config["epsilon_range"], attack_config["num_epsilons"]
        )
    else:
        epsilons = None

    # run attack
    accuracies = attack(resnet, resnet_loader, epsilons)

    print(
        f"\nAccuracies under {attack_config['attack_type']} attack: ", accuracies, "\n"
    )
    return accuracies


wandb_log = False

if __name__ == "__main__":
    parser = runtime_parser("attack_resnet")
    data_dir, save_dir, train_config, task_config, reg_config, attack_config = (
        get_configs("attack_resnet", parser.parse_args())
    )
    # load_dotenv()
    if wandb_log:
        # wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.login(key="8f078b070b04444ee9cd9d7f8c19f2d166a91aa4")
        wandb.init(
            # set the wandb project where this run will be logged
            project="Adversarial Attacks",
            reinit=True,
            # track hyperparameters and run metadata
            config={**train_config, **task_config, **reg_config, **attack_config},
        )
        wandb.define_metric("Perturbation_Strength")
        # define which metrics will be plotted against it
        wandb.define_metric("Accuracy", step_metric="Perturbation_Strength")

    path = attack_config["path_to_training_prints"]
