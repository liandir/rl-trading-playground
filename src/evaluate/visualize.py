"""Visualize utilities for evaluation and visualization helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CrosshairTool, DatetimeTickFormatter, HoverTool
from bokeh.plotting import figure, show


OHLCV_COLUMNS = ("time", "open", "high", "low", "close", "volume")


def visualize_ohlcv(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    title: str = "OHLCV",
    width: int = 1000,
    height: int = 600,
    show_plot: bool = False,
):
    """Create an interactive Bokeh candlestick chart with a volume subplot.

    Args:
        data (Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]]): The data value.
        title (str): The title value. Defaults to ``'OHLCV'``.
        width (int): The width value. Defaults to ``1000``.
        height (int): The height value. Defaults to ``600``.
        show_plot (bool): The show plot value. Defaults to ``False``.

    Returns:
        Any: The computed or requested result.
    """
    source = ColumnDataSource(_to_column_data(data))

    price_height = max(int(height * 0.72), 250)
    volume_height = max(height - price_height, 140)
    candle_width_ms = _infer_candle_width_ms(source.data["time"])

    price = figure(
        title=title,
        x_axis_type="datetime",
        width=width,
        height=price_height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    price.segment("time", "high", "time", "low", source=source, color="#2f3a45")
    price.vbar(
        "time",
        candle_width_ms,
        "body_top",
        "body_bottom",
        source=source,
        fill_color="color",
        line_color="color",
        alpha=0.9,
    )

    volume = figure(
        x_axis_type="datetime",
        x_range=price.x_range,
        width=width,
        height=volume_height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    volume.vbar(
        "time",
        candle_width_ms,
        0,
        "volume",
        source=source,
        fill_color="color",
        line_color="color",
        alpha=0.45,
    )

    _style_time_axis(price)
    _style_time_axis(volume)
    price.xaxis.visible = False
    price.yaxis.axis_label = "Price"
    volume.yaxis.axis_label = "Volume"

    hover = HoverTool(
        tooltips=[
            ("time", "@time{%F %T}"),
            ("open", "@open{0,0.0000}"),
            ("high", "@high{0,0.0000}"),
            ("low", "@low{0,0.0000}"),
            ("close", "@close{0,0.0000}"),
            ("volume", "@volume{0,0.00}"),
        ],
        formatters={"@time": "datetime"},
        mode="vline",
    )
    price.add_tools(hover, CrosshairTool(dimensions="height"))
    volume.add_tools(CrosshairTool(dimensions="height"))

    layout = column(price, volume, sizing_mode="fixed")
    if show_plot:
        show(layout)

    return layout


def _to_column_data(
    data: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
) -> dict[str, list[Any]]:
    """Convert the value to column data.

    Args:
        data (Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]]): The data value.

    Returns:
        dict[str, list[Any]]: The computed or requested result.
    """
    if hasattr(data, "to_dict"):
        columns = data.to_dict(orient="list")
    elif isinstance(data, Mapping):
        columns = {key: list(values) for key, values in data.items()}
    else:
        columns = _rows_to_columns(data)

    missing = [column for column in OHLCV_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")

    rows = len(columns["time"])
    if rows == 0:
        raise ValueError("OHLCV data is empty")

    result = {column: list(columns[column]) for column in OHLCV_COLUMNS}
    result["body_top"] = [
        max(open_, close) for open_, close in zip(result["open"], result["close"])
    ]
    result["body_bottom"] = [
        min(open_, close) for open_, close in zip(result["open"], result["close"])
    ]
    result["color"] = [
        "#16a34a" if close >= open_ else "#dc2626"
        for open_, close in zip(result["open"], result["close"])
    ]
    return result


def _rows_to_columns(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Rows to columns for evaluation and visualization helpers.

    Args:
        rows (Iterable[Mapping[str, Any]]): The rows value.

    Returns:
        dict[str, list[Any]]: The computed or requested result.
    """
    columns = {column: [] for column in OHLCV_COLUMNS}
    for row in rows:
        for column in OHLCV_COLUMNS:
            columns[column].append(row[column])
    return columns


def _infer_candle_width_ms(times: Sequence[Any]) -> int:
    """Infer candle width ms for evaluation and visualization helpers.

    Args:
        times (Sequence[Any]): The times value.

    Returns:
        int: The computed or requested result.
    """
    if len(times) < 2:
        return 60_000

    deltas = []
    for previous, current in zip(times, times[1:]):
        previous_ms = _time_to_ms(previous)
        current_ms = _time_to_ms(current)
        if current_ms > previous_ms:
            deltas.append(current_ms - previous_ms)

    if not deltas:
        return 60_000

    return max(int(min(deltas) * 0.7), 1)


def _time_to_ms(value: Any) -> float:
    """Time to ms for evaluation and visualization helpers.

    Args:
        value (Any): The value value.

    Returns:
        float: The computed or requested result.
    """
    if isinstance(value, datetime):
        return value.timestamp() * 1000
    return float(value)


def _style_time_axis(plot) -> None:
    """Style time axis for evaluation and visualization helpers.

    Args:
        plot (Any): The plot value.

    Returns:
        None: This function does not return a value.
    """
    plot.xaxis.formatter = DatetimeTickFormatter(
        seconds="%H:%M:%S",
        minsec="%H:%M:%S",
        minutes="%H:%M",
        hourmin="%H:%M",
        hours="%Y-%m-%d %H:%M",
        days="%Y-%m-%d",
        months="%Y-%m",
        years="%Y",
    )
    plot.grid.grid_line_alpha = 0.25
