# Copyright (C) 2020, Jiameng Fan <jmfan@bu.edu>
#
# This program is licenced under the MIT License,
# contained in the LICENCE file in this directory.


from torch.autograd import grad


def flat_grad(model, loss):
    """
    Get flat gradient given model and loss objective
    """
    # obtain gradients for each parameter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    grads = grad(loss, model_parameters, create_graph=True)[0]

    return grads.view(-1,)
