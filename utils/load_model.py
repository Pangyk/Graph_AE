import torch.nn.functional as F
import torch

def load_model_result(model, train_set2, valid_set, device):
    x1, e1, batch1 = None, None, None
    for data in train_set2:
        data = data.to(device)
        _, x1, e1, batch1 = model(data)
    # mean = torch.mean(x1, dim=0)
    # std = torch.std(x1, dim=0) + 1e-12
    # x1 = (x1 - mean) / std

    x2, e2, batch2 = None, None, None
    for data in valid_set:
        data = data.to(device)
        _, x2, e2, batch2 = model(data)
    # x2 = (x2 - mean) / std

    return [x1.detach(), e1.detach(), batch1.detach()], \
           [x2.detach(), e2.detach(), batch2.detach()]
