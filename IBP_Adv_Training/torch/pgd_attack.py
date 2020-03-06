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

    def perturb(self, data_nat, labels, epsilon=None, layer_idx=0):
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
                eta = self.a * data.grad.data.sign()
                data = data.detach().clone() + eta
                try:
                    eta = torch.clamp(data - data_nat, -eps, eps)
                except TypeError:
                    eta = torch.max(torch.min(data - data_nat, eps), -eps)
                data = torch.clamp(data_nat + eta, 0.0, 1.0)

            self.model.zero_grad()

            return data
