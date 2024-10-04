from models import ResNet
import time, torch, os, foolbox, utils, wandb
from foolbox import accuracy, PyTorchModel
import foolbox.attacks as fa
from adversarialAttacks.adversarialBase import AdversarialAttack
import random
import numpy as np


class BoundaryAttack(AdversarialAttack):
    def __init__(self, steps, task, log: bool = False, device: str = "cpu"):
        super().__init__(log=log, task=task, device=device)
        self.attack = fa.BoundaryAttack(steps=steps)
        self.name = "Decision-Based Boundary Attack"
        self.num_repeats = 5
        self.num_images = 1000

    def unit_scale(self, X):
        "scales the images between 0 and 1"
        return (X - self.min_x) / (self.max_x - self.min_x)

    def original_scale(self, X):
        "rescales the images between min_x and max_x"
        return X * (self.max_x - self.min_x) + self.min_x

    def get_adversarial(self, model, X, Y, epsilons):
        pass

    def get_median_distance(self, differences):
        "returns the median squared L2-distance (per dimension) across all samples"

        N_dim = differences.shape[-1]
        norm = np.linalg.norm(np.array(differences), axis=1)
        norm_squared = norm**2
        mean_across_dimensions = norm_squared / N_dim

        median = np.median(mean_across_dimensions)
        return median

    def get_attack_difference(self, fmodel, X):
        differences = []
        c = 0
        for i in range(len(X)):

            with torch.no_grad():
                # get initial prediction
                prediction = torch.tensor([fmodel(X[i : i + 1]).argmax()]).to(
                    self.device
                )

                try:
                    # apply attack
                    _, clipped_adv, success = self.attack(
                        fmodel, X[i : i + 1], prediction, epsilons=None
                    )
                    # if attack is successful. i.e prediction on adversarial image flips predicted label:
                    if success.float().sum().item():
                        im_np = X[i][0].cpu().detach().numpy()
                        adv_np = clipped_adv[0][0].cpu().detach().numpy()

                        im_original = self.unit_scale(im_np).flatten()
                        im_adversarial = self.unit_scale(adv_np).flatten()

                        difference = im_adversarial - im_original
                        differences.append(difference)

                except Exception as e:
                    print(f"Error for image {i}: {e}")
        return np.array(differences)

    def apply_attack_many_times(self, fmodel, X, num_repeats=5):
        medians = []
        for repeat in range(num_repeats):
            differences = self.get_attack_difference(fmodel, X)
            median = self.get_median_distance(differences)
            medians.append(median)

            print(
                f"\nRepeat: {repeat}. \nSquared median L2 distance between adversarial and original image per pixel: {median}\n"
            )

        medians = np.array(medians)

        return medians

    def __call__(self, model, loader, epsilons):
        "performing decision-based boundary attack via foolbox"

        X, Y = utils.get_images_from_loader(loader)
        X, Y = utils.preprocess(
            task=self.task, dataset=self.task, batch=(X, Y), device=self.device
        )
        X, Y = X.to(self.device), Y.to(self.device)

        self.min_x = X.min().item()
        self.max_x = X.max().item()
        model = model.to(self.device)

        selection_of_images = random.sample(range(len(Y)), self.num_images)
        X = X[selection_of_images].to(self.device)
        Y = Y[selection_of_images].to(self.device)

        fmodel = PyTorchModel(
            model, bounds=(self.min_x, self.max_x), device=self.device
        )

        all_medians = self.apply_attack_many_times(
            fmodel, X, num_repeats=self.num_repeats
        )

        median = np.mean(all_medians)
        if self.log:
            log_dict = {
                f"{self.name} - Median": median,
            }
            wandb.log(log_dict)

        return median, all_medians
