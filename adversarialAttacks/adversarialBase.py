import torch
from abc import abstractmethod


class AdversarialAttack:
    @abstractmethod
    def __init__(self, task: str, log: bool = False, device: str = "cpu"):
        self.log = log
        self.device = device
        self.task = task
    
    @abstractmethod
    def attack(self, model, X, Y, epsilons) -> torch.tensor:
        "Attacks model"
        raise NotImplementedError

    @abstractmethod
    def get_adversarial(self, model, X, Y, epsilon) -> torch.tensor:
        "Returns adversarial examples"
        raise NotImplementedError

    @abstractmethod
    def __call__(self, model, X, Y, epsilons) -> torch.tensor:
        raise NotImplementedError
