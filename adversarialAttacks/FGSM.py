import time, torch, os, foolbox, utils

# from foolbox import accuracy, PyTorchModel
from adversarialAttacks.adversarialBase import AdversarialAttack
from utils import timer_func


class FGSM(AdversarialAttack):
    def __init__(self, task, device = "cpu"):
        super().__init__(task= task, device = device)
        self.loss = torch.nn.CrossEntropyLoss()
        
    def preprocess(self, X):
        "scales the images between 0 and 1"
        min_x = X.min()
        max_x = X.max()
        self.min_x = min_x
        self.max_x = max_x
        return (X - min_x) / (max_x - min_x)

    def rescale(self, X):
        "rescales the images between min_x and max_x"
        return X * (self.max_x - self.min_x) - self.min_x

    def get_gradient_sign(self, model, X, Y) -> torch.tensor:
        "returns the sign of the gradient of the loss w.r.t the input (at each pixel)"
        gradient_signs = []
        for i in range(len(X)):
            data = X[i : i + 1].to(self.device)
            target = Y[i : i + 1].to(self.device)
            data.requires_grad = True
            output = model(data)
            loss = self.loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad_sign = data.grad.data.sign()
            gradient_signs.append(data_grad_sign)
        return torch.stack(gradient_signs).squeeze(2)

    def get_adversarial(self, model, X, Y, epsilon) -> torch.tensor:
        gradient_signs = self.get_gradient_sign(model, X, Y)
        X_preprocessed = self.preprocess(X)
        X_adv_not_scaled = X_preprocessed + epsilon * gradient_signs
        X_adv = self.rescale(X_adv_not_scaled)

        return X_adv

    @timer_func
    def __call__(self, model, X, Y, epsilons) -> list[float]:
        "Runs attack on model and returns accuracies. Using foolbox for speed."
        accuracies = []
        n = len(Y)
        for epsilon in epsilons:

            with torch.no_grad():
                X_adv = self.get_adversarial(model, X, Y, epsilon)
                preds = model(X_adv).argmax(axis=1)
                acc = (preds == Y).sum() / n

            accuracies.append(acc.item())
        return acc
