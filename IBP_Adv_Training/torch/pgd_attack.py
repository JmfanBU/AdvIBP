import torch
from torch.nn import CrossEntropyLoss


class LinfPGDAttack(object):
    def __init__(self, model, epsilon, k, a, random_start, loss_func):
        """
        Attack parameter initialization. The attack performs k steps of size a,
        while always staying within epsilon from the initial
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_func = loss_func

    def loss(self, output, labels):
        if self.loss_func == 'ce':
            loss = CrossEntropyLoss()(output, labels)
        else:
            raise ValueError("loss function has not been implemented.")
        return loss

    def perturb(self, data_nat, labels, layer_idx=0, c_t=None, epsilon=None):
        batch_size = data_nat.size(0)
        if c_t is not None:
            one_vec = torch.ones(batch_size).cuda()
        with torch.enable_grad():
            if epsilon is not None:
                eps = epsilon
            else:
                eps = self.epsilon

            if self.rand:
                try:
                    data = (
                        data_nat.detach().clone() +
                        torch.empty(data_nat.size()).uniform_(-eps, eps).cuda()
                    )
                except TypeError:
                    eps = torch.from_numpy(eps).cuda()
                    data = data_nat.detach().clone() + \
                        torch.rand_like(eps) * 2 * eps - eps
            else:
                data = data_nat.detach().clone()

            if self.k == 1:
                # FGSM attack
                self.a = 1.25 * eps

            for idxStep in range(self.k):
                if data.grad is not None:
                    data.grad.zero_()
                data.requires_grad_()
                output = self.model(
                    data, method_opt="forward", disable_multi_gpu=False,
                    layer_idx=layer_idx
                )
                self.model.zero_grad()
                loss = self.loss(output, labels)
                loss.backward()
                # data mask given inner maximization convergence
                if c_t is not None:
                    eta = self.a * one_vec[:, None, None, None] * \
                        data.grad.data.sign()
                else:
                    eta = self.a * data.grad.data.sign()
                data = data.detach().clone() + eta
                try:
                    eta = torch.clamp(data - data_nat, -eps, eps)
                except TypeError:
                    eta = torch.max(torch.min(data - data_nat, eps), -eps)
                data = torch.clamp(data_nat + eta, 0.0, 1.0)
                if c_t is not None:
                    # Evaluation of the inner maximization
                    c_x = self.FOSC(data, data_nat, eps)
                    one_vec[c_x <= c_t] = 0.
                    if one_vec.sum() == 0:
                        break

            self.model.zero_grad()
            if c_t is not None:
                return data, c_x.mean()
            else:
                return data

    def FOSC(self, data, data_nat, eps):
        c_x = eps * data.grad.data.norm(p='nuc', dim=(2, 3)) - \
            (data - data_nat) @ data.grad.data
        return c_x
