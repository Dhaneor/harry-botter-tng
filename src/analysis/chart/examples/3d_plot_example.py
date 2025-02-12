import plotly.graph_objects as go
import numpy as np

def create_5d_visualization(x, y, z, color, size):
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f'Color: {c}, Size: {s}' for c, s in zip(color, size)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='5D Visualization',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        width=800,
        height=800,
    )

    fig.show()

# Generate sample 5D data
n = 1000
x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)
color = np.random.rand(n)  # 4th dimension
size = np.random.rand(n) * 20  # 5th dimension

create_5d_visualization(x, y, z, color, size)