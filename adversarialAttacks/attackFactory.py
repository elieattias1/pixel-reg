from adversarialAttacks.adversarialBase import AdversarialAttack
from adversarialAttacks.transferredFGSM import TransferredFGSM
from adversarialAttacks.randomNoise import RandomAttack
from adversarialAttacks.BoundaryAttack import BoundaryAttack
from adversarialAttacks.FGSM import FGSM
from enum import Enum


class AttackType(Enum):
    TransferredFGSM = "TransferredFGSM"
    DecisionBoundary = "BoundaryAttack"
    Gaussian = "Gaussian"
    Uniform = "Uniform"
    SaltPepper = "SaltPepper"
    FGSM = "FGSM"


def AttackFactory(
    attack_type: str,
    task: str,
    wandb_log: bool,
    data_dir: str = "../data",
    save_dir: str = "../save",
    device: str = "cpu",
) -> AdversarialAttack:
    if attack_type == AttackType.TransfreredFGSM.value:
        return TransferredFGSM(
            task=task,
            log=wandb_log,
            data_dir=data_dir,
            save_dir=save_dir,
            device=device,
        )
    elif attack_type == AttackType.DecisionBoundary.value:
        return BoundaryAttack(steps=50, task=task, log=wandb_log, device=device)
    elif attack_type in [
        AttackType.Gaussian.value,
        AttackType.Uniform.value,
        AttackType.SaltPepper.value,
    ]:
        return RandomAttack(attack_type, task=task, log=wandb_log, device=device)
    elif attack_type == AttackType.FGSM.value:
        return FGSM(log=wandb_log, device=device, task=task)
    else:
        raise ValueError(f"Attack type {attack_type} not recognized")
