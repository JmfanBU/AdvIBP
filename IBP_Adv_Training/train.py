import sys
import copy
import numpy as np
import torch.optim as optim

from IBP_Adv_Training.models.bound_layers import BoundSequential, \
    BoundDataParallel
from IBP_Adv_Training.torch.training import Train
from IBP_Adv_Training.utils.eps_scheduler import EpsilonScheduler
from IBP_Adv_Training.utils.config import load_config, get_path, update_dict, \
    config_modelloader, config_dataloader
from IBP_Adv_Training.utils.argparser import argparser


class Logger(object):
    def __init__(self, log_file=None):
        self.log_file = log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file=self.log_file)
            self.log_file.flush()


def model_train(config, train_config, model, model_id, model_config):
    if "traininig_params" in model_config:
        train_config = update_dict(train_config,
                                   model_config["training_params"])
    model = BoundSequential.convert(
        model, train_config["method_params"]["bound_opts"]
    )

    # read traininig parameters from config file
    epochs = train_config["epochs"]
    lr = train_config["lr"]
    weight_decay = train_config["weight_decay"]
    starting_epsilon = train_config["starting_epsilon"]
    end_epsilon = train_config["epsilon"]
    schedule_start = train_config["schedule_start"]
    schedule_length = train_config["schedule_length"]
    optimizer = train_config["optimizer"]
    method = train_config["method"]
    verbose = train_config["verbose"]
    lr_decay_step = train_config["lr_decay_step"]
    lr_decay_milestones = train_config["lr_decay_milestones"]
    lr_decay_factor = train_config["lr_decay_factor"]
    multi_gpu = train_config["multi_gpu"]
    # parameters for the training method
    method_params = train_config["method_params"]
    # paramters for attack params
    attack_params = config["attack_params"]
    # parameters for evaluation
    evaluation_params = config["eval_params"]
    norm = float(train_config["norm"])
    train_data, test_data = config_dataloader(
        config, **train_config["loader_params"]
    )

    if optimizer == 'adam':
        opt = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer == 'sgd':
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9,
            nesterov=True, weight_decay=weight_decay
        )
    else:
        raise ValueError("Unknown optimizer")

    batch_multiplier = train_config["method_params"].get(
        "batch_multiplier", 1
    )
    batch_size = train_data.batch_size * batch_multiplier
    num_steps_per_epoch = int(
        np.ceil(1.0 * len(train_data.dataset) / batch_size)
    )
    epsilon_scheduler = EpsilonScheduler(
        train_config.get("schedule_type", "linear"),
        schedule_start * num_steps_per_epoch,
        ((schedule_start + schedule_length) - 1) * num_steps_per_epoch,
        starting_epsilon,
        end_epsilon,
        num_steps_per_epoch
    )
    max_eps = end_epsilon

    if lr_decay_step:
        # Use StepLR. Decay by lr_decay_factor every lr_decay_step.
        lr_scheduler = optim.lr_scheduler.StepLR(
            opt, step_size=lr_decay_step, gamma=lr_decay_factor
        )
    elif lr_decay_milestones:
        # Decay learning rate by lr_decay_factor at a few milestones
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=lr_decay_milestones, gamma=lr_decay_factor
        )
    else:
        raise ValueError(
            "one of lr_decay_step and"
            "lr_decay_milestones must be not empty."
        )
    model_name = get_path(config, model_id, "model", load=False)
    best_model_name = get_path(config, model_id, "best_model", load=False)
    model_log = get_path(config, model_id, "train_log")
    logger = Logger(open(model_log, "w"))
    logger.log(model_name)
    logger.log("Command line: ", " ".join(sys.argv[:]))
    logger.log("training configurations: ", train_config)
    logger.log("Model structure: ")
    logger.log(str(model))
    logger.log("data std: ", train_data.std)

    if multi_gpu:
        logger.log("\nUsing multiple GPUs for computing IBP bounds\n")
        model = BoundDataParallel(model)
    model = model.cuda()
    Train(
        model, model_id, model_name, best_model_name,
        epochs, train_data, test_data, multi_gpu,
        schedule_start, schedule_length,
        lr_scheduler, lr_decay_step, lr_decay_milestones,
        epsilon_scheduler, max_eps, norm, logger, verbose,
        opt, method, method_params, attack_params, evaluation_params
    )


def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    models, model_names = config_modelloader(config)
    for model, model_id, model_config in zip(models, model_names,
                                             config["models"]):
        # make a copy of global training config, and update per-model config
        train_config = copy.deepcopy(global_train_config)
        model_train(config, train_config, model, model_id, model_config)


if __name__ == "__main__":
    args = argparser()
    main(args)