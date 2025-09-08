from bokeh.plotting import figure, show
from bokeh.resources import CDN
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.palettes import Category10


def make_bokeh_plot(V_hist, w_hist, p_hist, C_hist, title="Portfolio Demo"):
    T = len(V_hist)
    N = len(w_hist[0])

    # Build sources
    t = list(range(T))
    source_V = ColumnDataSource(data=dict(t=t, V=V_hist, C=C_hist))
    figs = []

    # Figure 1: Portfolio value and cash
    f1 = figure(title=f"{title} — Portfolio Value", x_axis_label="t", y_axis_label="USD", width=900, height=320, tools="pan,wheel_zoom,box_zoom,reset,save,hover")
    r1 = f1.line('t', 'V', source=source_V, line_width=2, legend_label="Total V")
    r2 = f1.line('t', 'C', source=source_V, line_width=1, line_dash="dashed", legend_label="Cash C")
    f1.legend.click_policy = "hide"
    f1.add_tools(HoverTool(renderers=[r1], tooltips=[("t", "@t"), ("V", "@V{0,0.00}")], mode="vline"))

    # Figure 2: Prices per asset
    f2 = figure(title=f"{title} — Prices", x_axis_label="t", y_axis_label="USD", width=900, height=320, tools="pan,wheel_zoom,box_zoom,reset,save,hover")
    palette = Category10[10]
    for i in range(N):
        pi = [p_hist[t][i] for t in range(T)]
        src_i = ColumnDataSource(data=dict(t=t, p=pi))
        f2.line('t', 'p', source=src_i, line_width=1.5, legend_label=f"p[{i}]", color=palette[i % len(palette)])
    f2.legend.click_policy = "hide"

    # Figure 3: Holdings per asset (units)
    f3 = figure(title=f"{title} — Units held", x_axis_label="t", y_axis_label="units", width=900, height=320, tools="pan,wheel_zoom,box_zoom,reset,save,hover")
    for i in range(N):
        wi = [w_hist[t][i] for t in range(T)]
        src_i = ColumnDataSource(data=dict(t=t, w=wi))
        f3.line('t', 'w', source=src_i, line_width=1.5, legend_label=f"w[{i}]", color=palette[(i+3) % len(palette)])
    f3.legend.click_policy = "hide"

    layout = column(f1, f2, f3)
    show(layout)