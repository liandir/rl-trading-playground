import torch
from datetime import datetime
PI = 3.141592653589793238462


def smooth(data, time, tau):
    new_data = []

    val, t = data[0].clone(), time[0]
    for ti, di in zip(time, data):
        dt = ti - t
        val += (1 - torch.exp(-dt / tau)) * (di - val)
        # val += (dt / tau) * (di - val)
        new_data.append(val.clone())
        t = ti

    return torch.stack(new_data)


def compute_time_vector(timestamp: float) -> torch.Tensor:
    time = datetime.fromtimestamp(timestamp)
    dateiso = time.isocalendar()

    # ISO week count for the year (guard against edge cases)
    total_weeks = datetime(time.year, 12, 28).isocalendar().week

    period_day = (time.hour + (time.minute + time.microsecond / 1e6) / 60.0) / 24.0
    period_week = (dateiso.weekday - 1 + period_day) / 7.0
    period_year = (dateiso.week - 1 + period_week) / max(total_weeks, 1)

    angles = torch.tensor([period_day, period_week, period_year], dtype=torch.float32)
    
    return torch.concat([torch.sin(2 * PI * angles), torch.cos(2 * PI * angles)])
