import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def heatmap(data, title=None, x_axis=False, y_axis=False):
    """
    Creates a simple heatmap visualization from a 2D array.
    
    Args:
        data: 2D numpy array or list of lists containing the values to plot
        title: Optional string for the plot title
        x_axis: Optional string for the x-axis label
        y_axis: Optional string for the y-axis label
    
    Returns:
        plotly.graph_objects.Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            colorscale="Viridis"
        )
    )

    if title:
        fig.update_layout(title=title)

    x_label = x_axis if x_axis else f"col ({data.shape[1]})"
    y_label = y_axis if y_axis else f"row ({data.shape[0]})"

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    return fig

