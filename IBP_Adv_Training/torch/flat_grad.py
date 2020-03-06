from torch.autograd import grad


def flat_grad(model, loss):
    """
    Get flat gradient given model and loss objective
    """
    # obtain gradients for each parameter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    grads = grad(loss, model_parameters, create_graph=True)[0]

    return grads.view(-1,)
