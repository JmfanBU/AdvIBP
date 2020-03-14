import torch
from torch.nn import CrossEntropyLoss
from IBP_Adv_Training.utils.config import device


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
        if c_t is not None:
            batch_size = data_nat.size(0)
            one_vec = torch.ones(batch_size).cuda(device)
        with torch.enable_grad():
            if epsilon is not None:
                eps = epsilon
            else:
                eps = self.epsilon

            if self.rand:
                try:
                    data = (
                        data_nat.detach().clone() +
                        torch.empty(data_nat.size()).uniform_(
                            -eps, eps
                        ).cuda(device)
                    )
                except TypeError:
                    eps = torch.from_numpy(eps).cuda(device)
                    data = data_nat.detach().clone() + \
                        torch.rand_like(eps) * 2 * eps - eps
            else:
                data = data_nat.detach().clone()

            if self.k == 1:
                # FGSM attack
                self.a = 1.25 * eps
            else:
                self.a = eps / 4

            for idxStep in range(self.k):
                self.model.zero_grad()
                if data.grad is not None:
                    data.grad.zero_()
                if idxStep == 0:
                    data.requires_grad_()
                    output = self.model(
                        data, method_opt="forward", disable_multi_gpu=False,
                        layer_idx=layer_idx
                    )
                    loss = self.loss(output, labels)
                    loss.backward()
                    data_grad = data.grad.data
                # data mask given inner maximization convergence
                if c_t is not None:
                    eta = self.a * one_vec[:, None, None, None] * \
                        data_grad.sign()
                else:
                    eta = self.a * data_grad.sign()
                data = data.detach().clone() + eta
                try:
                    eta = torch.clamp(data - data_nat, -eps, eps)
                except TypeError:
                    eta = torch.max(torch.min(data - data_nat, eps), -eps)
                data = torch.clamp(data_nat + eta, 0.0, 1.0)

                data.requires_grad_()
                output = self.model(
                    data, method_opt="forward", disable_multi_gpu=False,
                    layer_idx=layer_idx
                )
                self.model.zero_grad()
                loss = self.loss(output, labels)
                loss.backward()
                data_grad = data.grad.data
                if c_t is not None:
                    # Evaluation of the inner maximization
                    c_x = self.FOSC(data, data_nat, eps)
                    one_vec[c_x <= c_t] = 0.
                    if one_vec.sum() == 0:
                        break
            c_x = self.FOSC(data, data_nat, eps)
            self.model.zero_grad()
            return data, c_x.mean()

    def FOSC(self, data, data_nat, eps):
        batch_size = data_nat.size(0)
        if torch.is_tensor(eps):
            c_x = (eps.view(batch_size, -1) *
                   data.grad.data.view(batch_size, -1)).norm(p=1, dim=1) - \
                ((data - data_nat).view(batch_size, -1) *
                 data.grad.data.view(batch_size, -1)).sum(dim=1)
        else:
            c_x = eps * \
                data.grad.data.view(batch_size, -1).norm(p=1, dim=1) - \
                ((data - data_nat).view(batch_size, -1) *
                 data.grad.data.view(batch_size, -1)).sum(dim=1)
        return c_x.view(-1)
