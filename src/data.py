import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple


PAIRS = {
    "Bitcoin":   "XBTEUR",
    "Ethereum":  "ETHEUR",
    "Ripple":    "XRPEUR",
    "Cardano":   "ADAEUR",
    "Polkadot":  "DOTEUR",
    "Chainlink": "LINKEUR",
    "Litecoin":  "LTCEUR",
    "Solana":    "SOLEUR",
    "Stellar":   "XLMEUR",
    "TRON":      "TRXEUR",
    "Monero":    "XMREUR",
    "Cosmos":    "ATOMEUR",
    "Dogecoin":  "DOGEEUR"
}
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def read_historical_data(path):
    data = []

    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) != 7 or parts[0] == 'time':
            continue
        timestamp = datetime.fromtimestamp(int(parts[0]))
        data.append({
            "time": timestamp,
            "open": float(parts[1]),
            "high": float(parts[2]),
            "low": float(parts[3]),
            "close": float(parts[4]),
            "volume": float(parts[5]),
            "trades": int(parts[6])
        })

    return data


def load_data(
        pairs,
        base_path: str = "../data/Kraken_OHLCVT/",
        interval: int = 1
    ):
    data = {}

    for name, symbol in pairs.items():
        try:
            print(f"Reading data for {name} ({symbol})...")
            file_name = f'{symbol}_{interval}.csv'
            d = read_historical_data(os.path.join(base_path, file_name))
            data[name] = sorted(d, key=lambda x: x['time'])
        except FileNotFoundError:
            print(f"Data file for {name} ({symbol}) not found.")

    return data


def align_data(data, interval):
    aligned_data, dt = {}, timedelta(minutes=interval)

    print("Determining common time range for alignment...")
    min_time = max([data[name][ 0]['time'] for name in data])
    max_time = min([data[name][-1]['time'] for name in data])
    print(f"Common time range: {min_time} to {max_time}")

    for name in data:
        print(f"Aligning data for {name}...")
        print(f"Total range for {name}: {data[name][0]['time']} to {data[name][-1]['time']}")
        filtered_data = [d for d in data[name] if d['time'] >= min_time and d['time'] <= max_time]

        new_data, times = [], []
        i, t = 0, min_time
        while t < max_time:
            if filtered_data[i]['time'] == t:
                new_data.append({**filtered_data[i].copy(), 'time': t})
                i += 1
            else:
                new_data.append({**filtered_data[i-1].copy(), 'time': t})
            times.append(t)
            t += dt
                
        aligned_data[name] = new_data

    return aligned_data, times


def align_data(
    data: Dict[str, List[dict]],
    interval_minutes: int,
    include_end: bool = False,
    allow_backfill: bool = False,   # if False, leave leading gaps as None until first point
) -> Tuple[Dict[str, List[dict]], List[datetime]]:
    """
    Aligns multiple time-ordered series of dicts (with 'time' keys) to a common grid.

    - Forward-fills between observed points.
    - Optionally backfills (use first point) before the first observation.
    - Computes a common range across series.
    - Safe against empty series or missing overlap.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")

    # Ensure each list is sorted and non-empty
    series = {}
    for name, rows in data.items():
        if not rows:
            series[name] = []
            continue
        series[name] = sorted(rows, key=lambda d: d["time"])

    # Determine common overlap
    starts = [rows[0]["time"] for rows in series.values() if rows]
    ends   = [rows[-1]["time"] for rows in series.values() if rows]
    if not starts or not ends:
        # no non-empty series
        return {name: [] for name in data}, []

    min_time = max(starts)
    max_time = min(ends)
    if min_time > max_time:
        # no overlap across series
        return {name: [] for name in data}, []

    dt = timedelta(minutes=interval_minutes)

    # Snap min_time up to the grid, max_time down/up depending on include_end
    def snap_up(t: datetime) -> datetime:
        delta = (t - datetime(t.year, t.month, t.day))
        offset = (delta.total_seconds() // (dt.total_seconds())) * dt
        snapped = datetime(t.year, t.month, t.day) + offset
        if snapped < t:
            snapped += dt
        return snapped

    def snap_down(t: datetime) -> datetime:
        delta = (t - datetime(t.year, t.month, t.day))
        offset = (delta.total_seconds() // (dt.total_seconds())) * dt
        return datetime(t.year, t.month, t.day) + offset

    grid_start = snap_up(min_time)
    grid_end   = snap_down(max_time)

    if grid_start > grid_end and not include_end:
        return {name: [] for name in data}, []

    # Build the grid once
    times: List[datetime] = []
    t = grid_start
    if include_end:
        while t <= grid_end:
            times.append(t)
            t += dt
    else:
        while t < grid_end:
            times.append(t)
            t += dt

    aligned: Dict[str, List[dict]] = {name: [] for name in data}
    if not times:
        return aligned, times

    # Align each series using a pointer and forward-fill
    for name, rows in series.items():
        if not rows:
            aligned[name] = [{**{}, "time": tt} for tt in times]  # or just []
            continue

        out: List[dict] = []
        i = 0
        last = None

        # advance i so that rows[i]['time'] <= times[0] if possible
        while i < len(rows) and rows[i]["time"] <= times[0]:
            last = rows[i]
            i += 1

        for tt in times:
            # Bring 'last' up to the most recent row at or before tt
            while i < len(rows) and rows[i]["time"] <= tt:
                last = rows[i]
                i += 1

            if last is None:
                # No past sample yet: either leave None or backfill from first row
                value = rows[0] if allow_backfill else None
            else:
                value = last

            # Create an aligned record; keep original fields if we have a value
            if value is None:
                out.append({"time": tt})
            else:
                out.append({**value, "time": tt})

        aligned[name] = out

    return aligned, times


def load_and_align_data(
        pairs,
        base_path: str = "../data/Kraken_OHLCVT/",
        interval: int = 1,
        include_end: bool = False,
        allow_backfill: bool = False
    ):
    data = load_data(pairs, base_path, interval)
    
    aligned_data, times = align_data(
        data, interval,
        include_end=include_end,
        allow_backfill=allow_backfill
    )
    
    return aligned_data, times


def get_field(data: list, field: str) -> float:
    """Convert a datetime to a float timestamp."""
    result = []
    for name in data:
        result.append([d[field] for d in data[name]])
    return result




