import time
from tqdm import tqdm
import copy
import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from IBP_Adv_Training.models.bound_layers import BoundLinear, BoundConv2d
from IBP_Adv_Training.torch.pgd_attack import LinfPGDAttack
from IBP_Adv_Training.torch.flat_grad import flat_grad
from IBP_Adv_Training.utils.eps_scheduler import EpsilonScheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Train(
    model, model_id, model_name, best_model_name,
    epochs, train_data, test_data, multi_gpu,
    schedule_start, schedule_length,
    lr_scheduler, lr_decay_step, lr_decay_milestones,
    epsilon_scheduler, max_eps, norm, logger, verbose,
    opt, method, method_params, attack_params, evaluation_params,
):
    # initialize logging values
    best_err = np.inf
    recorded_clean_err = np.inf
    timer = 0.0
    last_layer = None

    for idxLayer, Layer in enumerate(model if not multi_gpu else model.module):
        if isinstance(Layer, BoundLinear) or isinstance(Layer, BoundConv2d):
            if last_layer is not None:
                for param in last_layer.parameters():
                    param.requires_grad = False
            last_layer = Layer
            if idxLayer > 0:
                epsilon_scheduler.init_value = epsilon_scheduler.final_value
            # Start training
            for t in range(epochs):
                epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
                epoch_end_eps = epsilon_scheduler.get_eps(t + 1, 0)
                logger.log("\n\n==========Training==========")
                logger.log("Training Stage at Layer {}".format(idxLayer))
                logger.log(
                    "Epoch {}, learning rate {}, "
                    "epsilon {:.6g} - {:.6g}".format(
                        t, lr_scheduler.get_lr(),
                        epoch_start_eps, epoch_end_eps
                    )
                )
                start_time = time.time()
                epoch_train(
                    model, t, train_data, epsilon_scheduler,
                    max_eps, norm, logger, verbose, True, opt, method,
                    layer_idx=idxLayer, **method_params, **attack_params
                )
                if lr_decay_step:
                    # Use stepLR. Note that we manually set up epoch number
                    # here, so the +1 offset.
                    lr_scheduler.step(epoch=max(
                        t - (schedule_start + schedule_length - 1) + 1, 0
                    ))
                elif lr_decay_milestones:
                    # Use MultiStepLR with milestones.
                    lr_scheduler.step()
                epoch_time = time.time() - start_time
                timer += epoch_time
                logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(
                    epoch_time, timer
                ))
                logger.log("\n==========Evaluating==========")
                with torch.no_grad():
                    # evaluate
                    evaluation_eps = evaluation_params["epsilon"]
                    err, clean_err = epoch_train(
                        model, t, test_data,
                        EpsilonScheduler("linear", 0, 0,
                                         evaluation_eps, evaluation_eps, 1),
                        max_eps, norm, logger, verbose, False, None, method,
                        **method_params, **evaluation_params
                    )

                    logger.log("Saving to ", model_name)
                    torch.save({
                        'state_dict': model.module.state_dict()
                        if multi_gpu else model.state_dict(),
                        'epoch': t,
                    }, model_name)

                    # save the best model after we reached schedule
                    if t >= (schedule_start + schedule_length):
                        if err <= best_err:
                            best_err = err
                            recorded_clean_err = clean_err
                            logger.log(
                                'Saving best model {} with error {}'.format(
                                    best_model_name, best_err
                                )
                            )
                            torch.save({
                                'state_dict': model.module.state_dict()
                                if multi_gpu else model.state_dict(),
                                'robust_err': err,
                                'clean_err': clean_err,
                                'epoch': t,
                            }, best_model_name)
                logger.log('Total Time: {:.4f}'.format(timer))
                logger.log('Model {}, best err {}, clean err {}'.format(
                    model_id, best_err, recorded_clean_err
                ))


def epoch_train(
    model, t, loader, eps_scheduler, max_eps, norm,
    logger, verbose, train, opt, method, layer_idx=0, **kwargs
):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    num_class = 10
    losses = AverageMeter()
    errors = AverageMeter()
    adv_errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    batch_time = AverageMeter()
    batch_multiplier = kwargs.get("batch_multiplier", 1)
    coeff1, coeff2 = 1, 0
    optimal = False

    if train:
        model.train()
    else:
        model.eval()
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype=np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa)
    batch_size = loader.batch_size * batch_multiplier
    if batch_multiplier > 1 and train:
        logger.log("Warning: Large batch training. The equivalent batch size "
                   "is {} * {} = {}.".format(
                       batch_multiplier, loader.batch_size, batch_size
                   ))
    # per-channel std and mean
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    model_range = 0.0
    end_eps = eps_scheduler.get_eps(t + 1, 0)
    if end_eps < np.finfo(np.float32).tiny:
        logger.log("eps {} close to 0, using natural training".format(end_eps))
        method = "natural"
    if kwargs["adversarial_training"]:
        attack = LinfPGDAttack(
            model, kwargs.get("epsilon", max_eps),
            kwargs["attack_steps"], kwargs["attack_stepsize"],
            kwargs["random_start"], kwargs["loss_func"]
        )

    pbar = tqdm(loader)
    for i, (data, labels) in enumerate(pbar):
        start = time.time()
        intermediate_eps_scheduler = copy.deepcopy(eps_scheduler)
        eps = eps_scheduler.get_eps(t, int(i // batch_multiplier))
        if train and i % batch_multiplier == 0:
            opt.zero_grad()
        # upper bound matrix mask
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - \
            torch.eye(num_class).type_as(data).unsqueeze(0)
        # remove ground truth itself
        I_c = (~(labels.data.unsqueeze(1) ==
                 torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = c[I_c].view(data.size(0), num_class - 1, num_class)
        # scatter matrix to avoid compute margin to itself
        sa_labels = sa[labels]
        # storing computed upper and lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)
        ub_s = torch.zeros(data.size(0), num_class)

        if kwargs["bounded_input"]:
            # provided data is from range 0 - 1
            if norm != np.inf:
                raise ValueError(
                    "Bounded input only makes sense for Linf perturbation."
                    "Please set the bounded_input option to false."
                )
            data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / std), data_max)
            data_lb = torch.max(data - (eps / std), data_min)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                # For other norms, eps will be used instead
                data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            data_ub = data_ub.cuda()
            data_lb = data_lb.cuda()
            labels = labels.cuda()
            c = c.cuda()
            sa_labels = sa_labels.cuda()
            lb_s = lb_s.cuda()
            ub_s = ub_s.cuda()

        # omit the regular cross entropy, since we use robust error
        if kwargs["adversarial_training"] and method != "natural":
            output = model(data, method_opt="forward",
                           disable_multi_gpu=(method == "natural"))
            if layer_idx != 0:
                layer_ub, layer_lb, _, _, _, _ = model(
                    norm=norm, x_U=data_ub, x_L=data_lb, eps=eps,
                    layer_idx=layer_idx, method_opt="interval_range",
                    intermediate=True
                )
                data_ub, data_lb = layer_ub, layer_lb
                layer_eps_scheduler, layer_center, epsilon = intermediate_eps(
                    intermediate_eps_scheduler, layer_ub, layer_lb
                )
                layer_eps = layer_eps_scheduler.get_eps(
                    t, int(i // batch_multiplier)
                )
                data_adv = attack.perturb(
                    layer_center, labels,
                    epsilon=layer_eps, layer_idx=layer_idx
                )
                output_adv = model(
                    data_adv, method_opt="forward", layer_idx=layer_idx,
                    disable_multi_gpu=(method == "natural")
                )
            else:
                data_adv = attack.perturb(data, labels,
                                          epsilon=eps, layer_idx=layer_idx)
                output_adv = model(data_adv, method_opt="forward",
                                   disable_multi_gpu=(method == "natural"))
            # lower bound for adv training
            regular_ce = CrossEntropyLoss()(output_adv, labels)
        else:
            output = model(data, method_opt="forward",
                           disable_multi_gpu=(method == "natural"))
            regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(),
                                 data.size(0))
        errors.update(torch.sum(
            torch.argmax(output, dim=1) != labels
        ).cpu().detach().numpy() / data.size(0), data.size(0))
        if kwargs["adversarial_training"] and method != "natural":
            adv_errors.update(torch.sum(
                torch.argmax(output_adv, dim=1) != labels
            ).cpu().detach().numpy() / data.size(0), data.size(0))
        # get range statistics
        model_range = output.max().detach().cpu().item() - \
            output.min().detach().cpu().item()

        if verbose or method != "natural":
            if kwargs["bound_type"] == "interval":
                ub, lb, _, _, _, _ = model(
                    norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c,
                    layer_idx=layer_idx, method_opt="interval_range"
                )
            else:
                raise RuntimeError("Unknown bound_type " +
                                   kwargs["bound_type"])
            lb = lb_s.scatter(1, sa_labels, lb)
            # upper bound for adv training
            robust_ce = CrossEntropyLoss()(-lb, labels)

        if method == "robust":
            if train:
                regular_grads = flat_grad(model, regular_ce)
                robust_grads = flat_grad(model, robust_ce)
                coeff1, coeff2, optimal = two_obj_gradient(regular_grads,
                                                           robust_grads)
                loss = coeff1 * regular_ce + coeff2 * robust_ce
                model.zero_grad()
            else:
                loss = coeff1 * regular_ce + coeff2 * robust_ce
        elif method == "natural":
            loss = regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if train:
            loss.backward()
            if i % batch_multiplier == 0 or i == len(loader) - 1:
                opt.step()

        losses.update(loss.cpu().detach().numpy(), data.size(0))

        if verbose or method != "natural":
            robust_ce_losses.update(
                robust_ce.cpu().detach().numpy(), data.size(0)
            )
            robust_errors.update(torch.sum(
                (lb < 0).any(dim=1)
            ).cpu().detach().numpy() / data.size(0), data.size(0))

        batch_time.update(time.time() - start)

        if train:
            pbar.set_description(
                'Epoch: {}, eps: {:.4f}, coeff1: {:.2f}, coeff2: {:.2f}, '
                'optimal: {}, R: {model_range:.2f}'.format(
                    t, eps, coeff1, coeff2, optimal,
                    model_range=model_range,
                )
            )
        else:
            pbar.set_description(
                'Epoch: {}, eps: {:.4f}, '
                'Robust loss: {rb_loss.val:.2f}, '
                'Err: {errors.val:.3f}, '
                'Rob Err: {robust_errors.val:.3f}'.format(
                    t, eps,
                    model_range=model_range, rb_loss=robust_ce_losses,
                    errors=errors, adv_errors=adv_errors,
                    robust_errors=robust_errors
                )
            )
    logger.log(
        '----------Summary----------\n'
        'Reguler loss: {re_loss.avg:.2f}, '
        'Robust loss: {rb_loss.avg:.2f}, '
        'Err: {errors.avg:.3f}, '
        'Adv Err: {adv_errors.avg:.3f}, '
        'Rob Err: {robust_errors.avg:.3f},  '
        'R: {model_range:.2f}'.format(
            loss=losses, errors=errors, model_range=model_range,
            robust_errors=robust_errors, adv_errors=adv_errors,
            re_loss=regular_ce_losses, rb_loss=robust_ce_losses
        )
    )
    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg


def intermediate_eps(eps_scheduler, layer_ub, layer_lb):
    assert (layer_lb <= layer_ub).all()
    # center of the intermediate layer set
    intermediate_center = (layer_lb + layer_ub) / 2
    epsilon = layer_ub - intermediate_center
    eps_scheduler.init_value = epsilon.detach().cpu().numpy()
    eps_scheduler.final_value = epsilon.detach().cpu().numpy()
    return eps_scheduler, intermediate_center, epsilon


def two_obj_gradient(grad1, grad2, normalized=True):
    dot = torch.dot(grad1, grad2)
    grad1_norm = grad1.norm()
    grad2_norm = grad2.norm()
    grad1_normalized = grad1 / grad1_norm
    grad2_normalized = grad2 / grad2_norm
    optimal = False
    if dot >= grad1_norm.pow(2) or dot >= grad2_norm.pow(2):
        if dot > 0:
            bisector = grad1_normalized + grad2_normalized
            bisector_norm = bisector.norm()
            coeff = 0.5 / bisector_norm.pow(2) * \
                (grad1_norm + grad2_norm + dot / grad1_norm + dot / grad2_norm)
            coeff1 = coeff / grad1_norm
            coeff2 = coeff / grad2_norm
        else:
            coeff1 = 1.
            if grad2_norm == 0:
                coeff2 = 0.
            elif grad1_norm == 0:
                coeff1 = 0.
                coeff2 = 1.
            else:
                coeff2 = -dot / grad2_norm.pow(2)
    else:
        coeff1 = (grad2_norm.pow(2) - dot) / \
            (grad1_norm.pow(2) + grad2_norm.pow(2) - 2 * dot)
        coeff2 = 1 - coeff1
        optimal = True
    if normalized:
        coeff1, coeff2 = normalize(coeff1, coeff2)
    return coeff1, coeff2, optimal


def normalize(coeff1, coeff2):
    sum = coeff1 + coeff2
    return coeff1 / sum, coeff2 / sum
