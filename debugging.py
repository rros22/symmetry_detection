import numpy as np
from rich.console import Console
from rich.table import Table

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