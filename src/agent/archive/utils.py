import torch


def get_optimizer(model, lr: float, optim: str = "SGD", m: float | None = None):
    if optim == "SGD":
        return torch.optim.SGD(model.parameters(), lr, momentum=m if m is not None else 0)
    
    elif optim == "Adam":
        return torch.optim.Adam(model.parameters(), lr)
    
    elif optim == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr)
    
    else:
        raise NotImplementedError("The requested optimizer name is not recognized.")