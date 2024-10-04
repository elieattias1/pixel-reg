# -*- coding: utf-8 -*-


import os, argparse
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def runtime_parser(code_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--save_dir", default="save")

    if code_name in ["train_resnet", "test_resnet", "attack_resnet"]:

        parser.add_argument("--basis_num", default=None, type=int)
        parser.add_argument(
            "--modulator_features", default=[10, 10], nargs="+", type=int
        )

    if code_name in ["train_resnet", "attack_resnet", "process_jobs"]:
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--ignore_existing", action="store_true")

    if code_name in ["train_resnet", "process_jobs", "attack_resnet"]:
        parser.add_argument("--disp_num", default=6, type=int)
        parser.add_argument("--save_period", default=1, type=int)

    if code_name in ["train_resnet", "attack_resnet"]:
        parser.add_argument("--task", default="CIFAR10")
        parser.add_argument("--rgb", type=str2bool, default=False)

        parser.add_argument("--select_criterion", default="random")
        parser.add_argument("--core_shared", action="store_true")
        parser.add_argument("--conditioning", default=None, nargs="+")
        parser.add_argument("--max_best", default=5, type=int)
        parser.add_argument(
            "--arctanh",
            type=str2bool,
            default=True,
            help="Enable or disable arctanh feature",
        )
        parser.add_argument("--reg_alpha", type=float)
        parser.add_argument("--reg_type", default="pixel")
        parser.add_argument("--reg_image_num", default=5000, type=int)
        parser.add_argument(
            "--no-reg_boosting",
            dest="reg_boosting",
            action="store_false",
            help="Disable reg_boosting",
        )
        parser.add_argument("--reg_thresh", default=0.6, type=float)
        parser.add_argument("--reg_thresh_2", default=None, type=float)
        parser.add_argument("--reg_boosting_val", default=1.0, type=float)
        parser.add_argument("--reg_range", default="all", type=str)
        parser.add_argument("--reg_data", default="none", type=str)
        parser.add_argument("--reg_grab", default=False, type=bool)
        parser.add_argument("--reg_batch_size", default=16, type=int)
        parser.add_argument("--reg_boost_val_1", default=1.0, type=float)
        parser.add_argument("--start_fine_tune", default=0, type=int)
        parser.add_argument("--fine_tune_epochs", default=0, type=int)

        parser.add_argument("--attack", default=True, type=bool)

        parser.add_argument("--archi", default="ResNet18")
        parser.add_argument(
            "--base_channels",
            default=64,
            type=int,
            help="base channel number for ResNet",
        )

        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--valid_portion", default=0.01, type=float)
        parser.add_argument("--resnet_batch_size", default=64, type=int)

        parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
        parser.add_argument(
            "--momentum", default=0.9, type=float, help="momentum in SGD"
        )
        parser.add_argument(
            "--weight_decay", default=0, type=float, help="weight decay"
        )
        parser.add_argument("--phase_num", default=5, type=int)
        parser.add_argument("--phase_duration", default=4, type=int)
        parser.add_argument("--decay_rate", default=0.3, type=float)
        parser.add_argument("--epoch_num", default=40, type=int, help="epoch number")

        parser.add_argument("--num_epsilons", default=10, type=int)

    if code_name in ["create_jobs", "process_jobs", "clean_jobs"]:
        parser.add_argument("--job_type", choices=["train_resnet"])
        parser.add_argument("--job_name", default="jobs")

    if code_name == "process_jobs":
        parser.add_argument("--process_num", default=1, type=int)
        parser.add_argument(
            "--max_wait", default=60, type=float, help="seconds of wait before each job"
        )
    if code_name == "clean_jobs":
        parser.add_argument(
            "--tolerance", default=12, type=float, help="hours since start"
        )

    return parser


def get_configs(code_name, args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    if not os.path.exists(data_dir):
        os.makedirs("data")

    assert os.path.exists(data_dir), "data folder does not exist"
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        print("save folder does not exist, will be created")
        os.makedirs(save_dir)

    if code_name == "prepare_datasets":
        scans = args.scans
        valid_num, split_num = args.valid_num, args.split_num

    if code_name in ["train_resnet", "test_resnet", "process_jobs", "attack_resnet"]:
        if args.device.startswith("cuda") and torch.cuda.is_available():
            if args.device == "cuda:all":
                if torch.cuda.device_count() > 1:
                    device = "cuda:all"
                else:
                    device = "cuda"
            else:
                device = args.device
        elif args.device.startswith("mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if code_name in ["train_resnet", "process_jobs", "attack_resnet"]:
        run_config = {
            "device": device,
            "disp_num": args.disp_num,
            "save_period": args.save_period,
            "ignore_existing": args.ignore_existing,
        }
    if code_name in ["test_resnet"]:
        run_config = {
            "device": device,
            "ignore_existing": args.ignore_existing,
        }

    if code_name in ["train_resnet", "test_resnet", "attack_resnet"]:

        if args.select_criterion != "random":
            assert len(args.select_criterion.split("_")) == 2
            score_type, score_order = args.select_criterion.split("_")
            assert score_type in ["selectivity"]
            assert score_order in ["low", "high"]
        if args.reg_alpha is None:
            args.reg_type = "None"
            args.reg_image_num = "None"
            args.select_criterion = "None"
            args.core_shared = "None"
            args.conditioning = "None"
            args.max_best = "None"
            args.reg_boosting = "None"
            args.reg_thresh = "None"
            args.reg_thresh_2 = "None"
            args.reg_range = "None"
            args.reg_data = "None"
            args.reg_batch_size = "None"
            args.reg_grab = "None"
            args.reg_boosting_val = "None"
            args.reg_alpha = False
            args.arctanh = "None"

        task_config = {
            "task": args.task,
            "rgb": args.rgb,
            "select_criterion": args.select_criterion,
            "core_shared": args.core_shared,
            "conditioning": args.conditioning,
            "max_best": args.max_best,
        }
        reg_config = {
            "reg_thresh": args.reg_thresh,
            "reg_thresh_2": args.reg_thresh_2,
            "reg_range": args.reg_range,
            "reg_data": args.reg_data,
            "reg_boosting": args.reg_boosting,
            "reg_alpha": args.reg_alpha,
            "reg_image_num": args.reg_image_num,
            "reg_type": args.reg_type,
            "reg_batch_size": args.reg_batch_size,
            "reg_grab": args.reg_grab,
            "reg_boost_val_1": args.reg_boost_val_1,
            "start_fine_tune": args.start_fine_tune,
            "fine_tune_epochs": args.fine_tune_epochs,
            "reg_boosting_val": args.reg_boosting_val,
            "arctanh": args.arctanh,
        }

        r_model_config = {
            "archi": args.archi,
            "base_channels": args.base_channels,
        }

        train_config = {
            "seed": args.seed if code_name == "train_resnet" else None,
            "valid_portion": args.valid_portion,
            "resnet_batch_size": args.resnet_batch_size,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "phase_num": args.phase_num,
            "phase_duration": args.phase_duration,
            "decay_rate": args.decay_rate,
            "epoch_num": args.epoch_num,
        }
        attack_config = {
            "attack": args.attack,
            "num_epsilons": args.num_epsilons,
        }

    if code_name in ["create_jobs", "process_jobs", "clean_jobs"]:
        job_config = {
            "job_type": args.job_type,
            "job_name": args.job_name,
        }
    if code_name == "process_jobs":
        process_config = {
            "process_num": args.process_num,
            "max_wait": args.max_wait,
        }
    if code_name == "clean_jobs":
        process_config = {
            "tolerance": args.tolerance,
        }

    if code_name == "prepare_datasets":
        return data_dir, scans, valid_num, split_num
    if code_name == "train_resnet":
        return (
            data_dir,
            save_dir,
            run_config,
            task_config,
            r_model_config,
            train_config,
            reg_config,
            attack_config,
        )

    if code_name == "attack_resnet":
        return data_dir, save_dir, train_config, task_config, reg_config, attack_config
    if code_name == "create_jobs":
        return data_dir, save_dir, job_config
    if code_name == "process_jobs":
        return data_dir, save_dir, run_config, job_config, process_config
    if code_name == "clean_jobs":
        return save_dir, job_config, process_config
