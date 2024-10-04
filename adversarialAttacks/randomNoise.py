import torch, utils, wandb
from foolbox import accuracy, PyTorchModel
from adversarialAttacks.adversarialBase import AdversarialAttack
from utils import timer_func
from foolbox import accuracy, PyTorchModel


class RandomAttack(AdversarialAttack):
    def __init__(
        self, random_noise_type: str, task: str, log: bool = False, device: str = "cpu"
    ):
        super().__init__(log=log, task=task, device=device)
        self.random_noise_type = random_noise_type
        self.min_x = 0.0
        self.max_x = 1.0

    def preprocess(self, X):
        "scales the images between 0 and 1"
        return (X - self.min_x) / (self.max_x - self.min_x)

    def rescale(self, X):
        "rescales the images between min_x and max_x"
        return X * (self.max_x - self.min_x) + self.min_x

    def get_adversarial(self, model, batch, epsilon):
        X = self.preprocess(X=batch[0])  # between -2.5 and 2.5
        # print("in get adversarial :", X.min(), X.max())
        Y = batch[1].to(self.device)

        X = X.to(self.device)

        X_adv_not_scaled = utils.noisy_inputs(
            inputs=X,
            noise_type=self.random_noise_type,
            epsilon=epsilon,
        )  # between 0 and 1

        # print("X_adv range", X_adv_not_scaled.min(), X_adv_not_scaled.max())

        X_adv = self.rescale(X_adv_not_scaled)  # between -2.5 and 2.5
        # print("X_adv rescaled:", X_adv.min(), X_adv.max())
        return X_adv, Y

    def __call__(
        self, model, loader, epsilons: torch.tensor, epoch_num=0, only_correct=False
    ):
        accuracies = []

        model = model.to(self.device)
        model.eval()

        self.min_x = float("inf")
        self.max_x = float("-inf")
        for epsilon in epsilons:
            total = 0  # Total number of samples processed for current epsilon
            correct = 0  # Total correct predictions for current epsilon

            for batch in loader:
                X, Y = utils.preprocess(
                    task=self.task,
                    dataset=self.task,
                    device=self.device,
                    batch=batch,
                )  # between -2.5  and 2.5
                X = X.to(self.device)
                Y = Y.to(self.device)

                # Update global min and max values
                self.min_x = min(self.min_x, X.min().item())
                self.max_x = max(self.max_x, X.max().item())

                # Filter to only include initially correctly classified examples if only_correct is True
                X_adv, Y_adv = self.get_adversarial(model, (X, Y), epsilon)
                X_adv, Y_adv = X_adv.to(self.device), Y_adv.to(self.device)

                with torch.no_grad():
                    outputs = model(X_adv)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(Y_adv).sum().item()
                    total += Y_adv.size(0)

            # Calculate accuracy for the current epsilon after processing all batches
            acc = correct / total if total > 0 else 0  # avoid division by zero
            accuracies.append(acc)

            print(f"Epsilon: {epsilon}. Total Accuracy: {acc}")

            if self.log:
                log_dict = {
                    f"{self.random_noise_type} - accuracy": acc,
                    "Perturbation_Strength": epsilon,
                    "Min": self.min_x,
                    "Max": self.max_x,
                }
                wandb.log(log_dict)

        return torch.tensor(accuracies).reshape(1, -1)
