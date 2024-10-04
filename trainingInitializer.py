from utils import get_images_from_loader, get_s_matrix, get_reg_loader
import utils
import random
from torch import optim


class TrainingInitializer:
    def __init__(
        self,
        data_dir,
        save_dir,
        run_config,
        task_config,
        r_model_config,
        train_config,
        reg_config,
        attack_config,
        device,
    ):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.run_config = run_config
        self.task_config = task_config
        self.r_model_config = r_model_config
        self.train_config = train_config
        self.reg_config = reg_config
        self.attack_config = attack_config
        self.device = device
        self.resnet_loader_train = None
        self.resnet_loader_valid = None
        self.resnet_loader_test = None
        self.loaders = None
        self.X_test = None
        self.Y_test = None
        self.X_rep = None
        self.image_indices = None
        self.all_reg_images = None
        self.labels = None
        self.s_matrix = None
        self.indices_tensor = None
        self.reg_loader = None
        self.image_idxs = None
        self.resnet = None
        self.optimizer = None

    def setup_dataloaders(self):
        self.resnet_loader_train, self.resnet_loader_valid, self.resnet_loader_test = (
            utils.create_resnet_loaders(
                self.data_dir, self.task_config, self.train_config
            )
        )

        self.loaders = [
            self.resnet_loader_train,
            self.resnet_loader_valid,
            self.resnet_loader_test,
        ]
        self.X_test, self.Y_test = get_images_from_loader(self.resnet_loader_test)
        self.X_test, self.Y_test = self.X_test.to(self.device), self.Y_test.to(
            self.device
        )

        # To measure how image representations change across learning
        X_rep, _ = get_images_from_loader(self.resnet_loader_train)
        num_images_for_representation = 1000
        self.image_indices = random.sample(
            range(len(X_rep)), num_images_for_representation
        )
        self.X_rep = X_rep[self.image_indices]

    def setup_regularization(self):
        if self.reg_config["reg_alpha"] is not None:
            self.all_reg_images, self.labels = utils.fetch_reg_data(
                self.data_dir,
                self.task_config,
                self.reg_config,
                num_reg_images=self.reg_config["reg_image_num"],
            )
            self.all_reg_images = self.all_reg_images.to(self.device)
            self.s_matrix, self.indices_tensor = get_s_matrix(
                self.reg_config, self.all_reg_images, self.labels
            )
            self.reg_loader, self.image_idxs = get_reg_loader(
                self.reg_config, self.indices_tensor, self.all_reg_images
            )
        else:
            print("No regularization")

    def setup_model_and_optimizer(self):
        self.resnet = utils.create_resnet_model(
            self.task_config, self.r_model_config
        ).to(self.device)
        self.optimizer = optim.SGD(
            self.resnet.parameters(),
            lr=0.0,
            momentum=self.train_config["momentum"],
            weight_decay=self.train_config["weight_decay"],
        )

    def initialize_training(self):
        self.setup_dataloaders()
        self.setup_regularization()
        self.setup_model_and_optimizer()
