import torch, os, wandb
from adversarialAttacks.adversarialBase import AdversarialAttack
from adversarialAttacks.FGSM import FGSM
from adversarialAttacks.ModelLibrary import ModelLibrary
import utils
from utils import timer_func


class TransferredFGSM(AdversarialAttack):
    def __init__(
        self,
        task: str,
        log: bool = False,
        data_dir: str = "data",
        save_dir: str = "save",
        device: str = "cpu",
    ):
        super().__init__(log=log, device=device, task=task)
        if task == "MNIST":
            self.training_id_none = ModelLibrary.MNIST.value
        elif task == "FashionMNIST":
            self.training_id_none = ModelLibrary.FASHION.value
        elif task == "CIFAR10":
            self.training_id_none = ModelLibrary.CIFAR10.value
        elif task == "CIFAR100":
            self.training_id_none = ModelLibrary.CIFAR100.value
        else:
            raise ValueError(f"Task {task} not recognized")
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.gradient_sign_path = f"{data_dir}/transfered_perturbations_{task}.pt"
        self.model_noreg = self.get_transfer_model().to(self.device)

    def get_transfer_model(self):
        resnet_noreg, _, _, _ = utils.load_model(
            save_dir=self.save_dir, training_id=self.training_id_none
        )
        _ = resnet_noreg.eval()

        resnet_noreg = resnet_noreg.to(self.device)

        return resnet_noreg

    def get_adversarial(
        self,
        model,
        X,
        Y,
        epsilon,
    ):

        mini = X.min()
        maxi = X.max()
        factor = maxi - mini

        if epsilon > 0:
            gradient_sign = (
                FGSM(task=self.task, device=self.device)
                .get_gradient_sign(self.model_noreg, X, Y)
                .to(self.device)
            )

            epsilon_rescaled = factor * epsilon
            X_adv = X + epsilon_rescaled * gradient_sign
        else:
            epsilon_rescaled = epsilon
            X_adv = X

        X_adv = torch.clamp(X_adv, mini, maxi)

        return X_adv, Y

    def __call__(self, model, loader, epsilons) -> list[float]:
        accuracies = []
        model = model.to(self.device)
        for epsilon in epsilons:
            correct = 0
            total = 0
            for batch in loader:
                X, Y = utils.preprocess(
                    task=self.task,
                    dataset=self.task,
                    device=self.device,
                    batch=batch,
                )
                X, Y = X.to(self.device), Y.to(self.device)

                X_adv, labels = self.get_adversarial(model, X, Y, epsilon)
                X_adv, labels = X_adv.to(self.device), labels.to(self.device)
                with torch.no_grad():

                    _, predicted = model(X_adv).max(1)
                    correct += predicted.eq(Y).sum().item()
                    total += len(Y)

            print(f"Accuracy for epsilon {epsilon} : {correct/total}")
            acc = correct / total
            accuracies.append(acc)
            if self.log:
                log_dict = {
                    f"TransferedFGSM - accuracy": acc,
                    "Perturbation_Strength": epsilon,
                }

                wandb.log(log_dict)

        return torch.tensor(accuracies)
