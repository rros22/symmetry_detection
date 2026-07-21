import numpy as np
from rich.console import Console
from rich.table import Table
from matplotlib import pyplot as plt

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
    print(f"  {'Domain (X Span)':<25} : [{x_start}, {x_end}]")
    print(f"  {'Points per Trajectory':<25} : {num_points}")
    print(f"  {'Initial Conditions':<25} : {ic_str}")
    print(f"  {'Solver Method':<25} : {method}")
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

def _unit_cube_transform(X, U, P, N, x_range, u_range, p_range):
    spans = np.array([
        x_range[1] - x_range[0],
        u_range[1] - u_range[0],
        p_range[1] - p_range[0],
    ])
    Xn = (X - x_range[0]) / spans[0]
    Un = (U - u_range[0]) / spans[1]
    Pn = (P - p_range[0]) / spans[2]
    Nn = N * spans[:, np.newaxis, np.newaxis]
    return Xn, Un, Pn, Nn


def _finish_3d_plot(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_zlabel('p')


def scaled_3D_quiver_surface(X_grid, U_grid, P_grid, N, x_range, u_range, p_range):
    """Plot an equation manifold surface with its normal field."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    Xn, Un, Pn, Nn = _unit_cube_transform(X_grid, U_grid, P_grid, N, x_range, u_range, p_range)
    ax.plot_surface(Xn, Un, Pn, cmap='coolwarm')
    ax.quiver(Xn, Un, Pn, Nn[0], Nn[1], Nn[2], normalize=True, length=0.08, color='k', alpha=0.3)
    _finish_3d_plot(ax)

    return fig, ax


def scaled_3D_quiver_trajectories(X, normal_fn, x_range, u_range, p_range, *, style="lines"):
    """
    Plot integrated trajectories with their normal field.

    X has shape (n_trajectories, 3, m_points) with rows (x, u, p).
    style: "lines" plots each trajectory as a curve; "scatter" plots all sample points.
    """
    if style not in ("lines", "scatter"):
        raise ValueError(f"style must be 'lines' or 'scatter', got {style!r}")

    x = X[:, 0]
    u = X[:, 1]
    p = X[:, 2]
    N = normal_fn(x, u)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    Xn, Un, Pn, Nn = _unit_cube_transform(x, u, p, N, x_range, u_range, p_range)
    if style == "lines":
        for i in range(Xn.shape[0]):
            ax.plot(Xn[i], Un[i], Pn[i], color='C0', linewidth=1.0)
    else:
        ax.scatter(Xn.ravel(), Un.ravel(), Pn.ravel(), color='C0', s=8, alpha=0.6)
    ax.quiver(Xn, Un, Pn, Nn[0], Nn[1], Nn[2], normalize=True, length=0.08, color='k', alpha=0.3)
    _finish_3d_plot(ax)

    return fig, ax