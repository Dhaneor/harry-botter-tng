import plotly.graph_objects as go
import numpy as np

def create_3d_heatmap(x, y, z, values):
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        opacity=0.1,
        surface_count=17,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title='3D Heatmap',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        width=700,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()

# Generate sample data
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
z = np.linspace(0, 1, 20)

X, Y, Z = np.meshgrid(x, y, z)

# This is your dependent variable
values = np.sin(X*10) + np.cos(Y*10) + Z

create_3d_heatmap(X, Y, Z, values)