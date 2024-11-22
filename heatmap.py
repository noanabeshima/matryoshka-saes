import plotly.graph_objects as go


def heatmap(data, title=None, dim_names=(None, None)):
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
        data=go.Heatmap(z=data, colorscale="Viridis"),
        layout=go.Layout(yaxis=dict(autorange="reversed"))
    )

    if title:
        fig.update_layout(title=title)

    row_label = f"{dim_names[0]} ({data.shape[0]})" if dim_names[0] is not None else f"row ({data.shape[0]})"
    col_label = f"{dim_names[1]} ({data.shape[1]})" if dim_names[1] is not None else f"col ({data.shape[1]})"

    fig.update_layout(
        xaxis_title=col_label,
        yaxis_title=row_label,
        height=600,
        width=1000,
    )

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    return fig
