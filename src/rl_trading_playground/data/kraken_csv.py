"""Kraken OHLCVT CSV data utilities for multi-currency trading.

This module contains the legacy CSV extraction helpers for Kraken-formatted
historical market files. Files are expected to be named with the pattern
``{symbol}_{interval}.csv`` and contain Unix timestamp, OHLC, volume, and trade
count columns.
"""
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Tuple


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


def read_historical_data(path: str):
    """Read historical OHLCVT rows from a Kraken CSV file.

    Args:
        path: Path to a Kraken OHLCVT CSV file. Rows with headers or malformed
            column counts are skipped.

    Returns:
        A list of dictionaries with ``time``, ``open``, ``high``, ``low``,
        ``close``, ``volume``, and ``trades`` fields.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a well-formed row contains a value that cannot be
            converted to the expected numeric type.
    """
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
        pairs: Mapping[str, str],
        base_path: str = "../data/Kraken_OHLCVT/",
        interval: int = 1
    ):
    """Load Kraken OHLCVT CSV data for the configured pairs.

    Args:
        pairs: Mapping of display names to Kraken pair symbols.
        base_path: Directory containing Kraken CSV files.
        interval: Interval suffix used in the expected filename pattern
            ``{symbol}_{interval}.csv``.

    Returns:
        A dictionary keyed by display name, with each value sorted by the
        ``time`` field. Missing files are reported to stdout and omitted from
        the result.
    """
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


def align_data(
    data: Dict[str, List[dict]],
    interval_minutes: int,
    include_end: bool = False,
    allow_backfill: bool = False,
) -> Tuple[Dict[str, List[dict]], List[datetime]]:
    """Align multiple time-ordered OHLCVT series to a common time grid.

    Args:
        data: Mapping of series names to row dictionaries containing a
            ``time`` key.
        interval_minutes: Grid interval in minutes. Must be greater than zero.
        include_end: Whether to include the final snapped timestamp in the
            output grid.
        allow_backfill: Whether to fill leading gaps with the first known row.
            If false, leading gaps are represented by a dictionary containing
            only ``time``.

    Returns:
        A tuple of ``(aligned, times)``. ``aligned`` is keyed like ``data`` and
        contains one row per timestamp in ``times``.

    Raises:
        ValueError: If ``interval_minutes`` is less than or equal to zero.
        KeyError: If any non-empty input row is missing the ``time`` key.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")

    series = {}
    for name, rows in data.items():
        if not rows:
            series[name] = []
            continue
        series[name] = sorted(rows, key=lambda d: d["time"])

    starts = [rows[0]["time"] for rows in series.values() if rows]
    ends = [rows[-1]["time"] for rows in series.values() if rows]
    if not starts or not ends:
        return {name: [] for name in data}, []

    min_time = max(starts)
    max_time = min(ends)
    if min_time > max_time:
        return {name: [] for name in data}, []

    dt = timedelta(minutes=interval_minutes)

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
    grid_end = snap_down(max_time)

    if grid_start > grid_end and not include_end:
        return {name: [] for name in data}, []

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

    for name, rows in series.items():
        if not rows:
            aligned[name] = [{**{}, "time": tt} for tt in times]
            continue

        out: List[dict] = []
        i = 0
        last = None

        while i < len(rows) and rows[i]["time"] <= times[0]:
            last = rows[i]
            i += 1

        for tt in times:
            while i < len(rows) and rows[i]["time"] <= tt:
                last = rows[i]
                i += 1

            if last is None:
                value = rows[0] if allow_backfill else None
            else:
                value = last

            if value is None:
                out.append({"time": tt})
            else:
                out.append({**value, "time": tt})

        aligned[name] = out

    return aligned, times


def load_and_align_data(
        pairs: Mapping[str, str],
        base_path: str = "../data/Kraken_OHLCVT/",
        interval: int = 1,
        include_end: bool = False,
        allow_backfill: bool = False
    ):
    """Load Kraken CSV data and align it to a common interval grid.

    Args:
        pairs: Mapping of display names to Kraken pair symbols.
        base_path: Directory containing Kraken CSV files.
        interval: Interval suffix used for loading files and the alignment grid
            size in minutes.
        include_end: Whether to include the final snapped timestamp in the
            output grid.
        allow_backfill: Whether to fill leading gaps with the first known row.

    Returns:
        A tuple of ``(aligned_data, times)`` from :func:`align_data`.
    """
    data = load_data(pairs, base_path, interval)

    aligned_data, times = align_data(
        data, interval,
        include_end=include_end,
        allow_backfill=allow_backfill
    )

    return aligned_data, times


def get_field(data: Mapping[str, List[Mapping[str, Any]]], field: str) -> list[list[Any]]:
    """Return one field across all named series.

    Args:
        data: Mapping of series names to row dictionaries.
        field: Field name to extract from each row.

    Returns:
        A list containing one list of extracted values per series, preserving
        the iteration order of ``data``.

    Raises:
        KeyError: If any row does not contain ``field``.
    """
    result = []
    for name in data:
        result.append([d[field] for d in data[name]])
    return result
