import numpy as np
from rich.console import Console
from rich.table import Table
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

# Printing for debbuging
def print_configuration(ode_name, x_start, x_end, initial_conditions, num_points, method):
    # Format array preview nicely
    ic_arr = np.asarray(initial_conditions)
    if len(ic_arr) > 4:
        ic_str = f"[{ic_arr[0]:.2f}, {ic_arr[1]:.2f}, ..., {ic_arr[-1]:.2f}] ({len(ic_arr)} trajectories)"
    else:
        ic_str = f"{np.round(ic_arr, 2).tolist()}"

    title = f" INTEGRATION CONFIGURATION: {ode_name.upper()} "
    width = 60

    print("=" * width)
    print(f"{title.center(width, ' ')}")
    print("=" * width)
    print(f"  {"Domain (X Span)":<25} : [{x_start}, {x_end}]")
    print(f"  {"Points per Trajectory":<25} : {num_points}")
    print(f"  {"Initial Conditions":<25} : {ic_str}")
    print(f"  {"Solver Method":<25} : {method}")
    print("-" * width)

def print_configuration_rich(ode_name, x_start, x_end, initial_conditions, num_points, method):
    console = Console()
    ic_arr = np.asarray(initial_conditions)
    
    table = Table(title=f"ODE Configuration — {ode_name.upper()}", title_style="bold cyan", border_style="dim")
    table.add_column("Parameter", style="bold white", justify="left")
    table.add_column("Value", style="green", justify="left")

    table.add_row("X Range", f"[{x_start}, {x_end}]")
    table.add_row("Resolution", f"{num_points} points/trajectory")
    table.add_row("Trajectory Count", f"{len(ic_arr)} initial conditions")
    table.add_row("IC Range", f"[{ic_arr.min():.2f} ... {ic_arr.max():.2f}]")
    table.add_row("Integrator", method)

    console.print(table)

# Plots

def _set_data_tick_labels(ax, x_range, u_range, p_range):
    """Show physical x, u, p values on axes that live in [0, 1]."""
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{x_range[0] + v * (x_range[1] - x_range[0]):.2g}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{u_range[0] + v * (u_range[1] - u_range[0]):.2g}")
    )
    ax.zaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{p_range[0] + v * (p_range[1] - p_range[0]):.2g}")
    )


def scaled_3D_quiver(X_grid, U_grid, P_grid, N, x_range, u_range, p_range):
    """
    Plot a surface with its normal field in (x, u, p) space.

    mplot3d quiver has no ``angles='xy'`` equivalent. Rescaling vector
    components in data space does not fix directions for the same reason that
    2D ``angles='uv'`` ignores axis scaling. The fix is to plot surface and
    normals in unit-cube coordinates [0, 1]^3 with ``set_box_aspect((1,1,1))``,
    then label ticks in physical units.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    spans = np.array([
        x_range[1] - x_range[0],
        u_range[1] - u_range[0],
        p_range[1] - p_range[0],
    ])

    # Scale points down to unit cube
    Xn = (X_grid - x_range[0]) / (x_range[1] - x_range[0])
    Un = (U_grid - u_range[0]) / (u_range[1] - u_range[0])
    Pn = (P_grid - p_range[0]) / (p_range[1] - p_range[0])

    # Scale vector field accordingly
    Nn = N* spans[:, np.newaxis, np.newaxis]
    print(Nn.shape)
    ax.plot_surface(Xn, Un, Pn, cmap='coolwarm')
    ax.quiver(
        Xn, Un, Pn,
        Nn[0], Nn[1], Nn[2],
        normalize=True,
        length=0.08,
    )

    # Set new axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Set aspect ratio to 1 so that the vectors look normal
    ax.set_box_aspect((1, 1, 1))
    # _set_data_tick_labels(ax, x_range, u_range, p_range)

    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_zlabel('p')

    return fig, ax