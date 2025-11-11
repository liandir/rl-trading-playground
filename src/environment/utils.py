import torch


def smooth(data, time, tau):
    new_data = []

    val, t = data[0].clone(), time[0].clone()
    for ti, di in zip(time, data):
        dt = ti - t
        val += (1 - torch.exp(-dt / tau)) * (di - val)
        new_data.append(val.clone())

    return torch.stack(new_data)
