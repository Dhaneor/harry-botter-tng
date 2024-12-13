#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product
import numpy as np

rows = 3
cols = 5
horizontal_spacing = 0.05
vertical_spacing = 0.1
legend_horizontal_spacing = -0.005  # vertical buffer between legend and subplot

subplot_height = (1 - (rows - 1) * vertical_spacing) / rows
subplot_width = (1 - (cols - 1) * horizontal_spacing) / cols

fig = make_subplots(
    rows=rows,
    cols=cols,
    vertical_spacing=vertical_spacing,
    horizontal_spacing=horizontal_spacing,
    # shared_yaxes=True,
    shared_xaxes=True,
)

for i, (row, col) in enumerate(
    product(range(1, rows + 1), range(1, cols + 1))
):  # rows & cols are 1-indexed
    x_data = np.arange(100)
    y_data = np.convolve(np.random.normal(0, 50, 100), np.ones(10), mode="same")
    fig.append_trace(
        go.Scatter(x=x_data, y=y_data, name=f"Trace {i}"), row=row, col=col
    )

    # Add second series to some subplots
    if np.random.choice(3) == 0:  # Condition to add a second series
        y_data2 = np.convolve(np.random.normal(0, 50, 100), np.ones(10), mode="same")
        fig.add_trace(
            go.Scatter(x=x_data, y=y_data2, name=f"Trace {i}b"), row=row, col=col
        )

    legend_name = (
        f"legend{i+2}"  # legend1 is the theme's default. start at legend2 to avoid.
    )
    x = ((col - 1) * (subplot_width + horizontal_spacing)) + (subplot_width / 2)
    y = (
        1
        - ((row - 1) * (subplot_height + vertical_spacing))
        + legend_horizontal_spacing
    )
    fig.update_traces(row=row, col=col, legend=legend_name)
    fig.update_layout(
        {
            legend_name: dict(
                x=x, y=y, xanchor="center", yanchor="bottom", bgcolor="rgba(0,0,0,0)"
            )
        }
    )


fig.update_layout(height=600, title_text="<b>Unique Legend Per Subplot Demonstration")

fig.show()
