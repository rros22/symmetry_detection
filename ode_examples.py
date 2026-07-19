"""
    - Three standard ODEs with point symmetries
    - Trajectory generation
"""
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

def bernoulli_ode(x,u):
     
     """
        Bernoulli equation du/dx = 2*u/x - u^2*x^2
     """

     return 2*u/x - (u**2)*(x**2)

def rational_ode(x,u):

    """
        Rational equation with rotation symmetry du/dx = (u^3 + u*x^3 - u - x)/(x*u^2 + x^3 + u - x)
    """

    return (u**3 + u*x**3 - u - x)/(x*u**2 + x**3 + u - x)

def riccati_ode(x,u):

    """
        Riccati equation du/dx = x*u^2 - 2*u/x - 1/x^3
    """

    return x*u**2 - 2*u/x - 1/x**3

def scaling_ode(x, u):

    """
        Non-linear ODE with scaling symmetry du/dx = u/x + x/u
    """

    return u/x + x/u

def abel_ode(x,u):

    """
        Abel equation of the first kind
    """

    return u**3 - x

ODES = {
    "bernoulli": bernoulli_ode,
    "rational": rational_ode,
    "riccati": riccati_ode,
    "scaling": scaling_ode,
    "abel": abel_ode
}

INTEGRATORS = {
    "RK45": "RK45",
    "RK23": "RK23",
    "DOP853": "DOP853",
    "Radau": "Radau",
    "BDF": "BDF",
    "LSODA": "LSODA",
}

# Generate a full trajectory

def generate_trajectory(ode_name, x_start=1, x_end=10, u0=1, num_points=30, method="RK45"):
    """
        The main programmatic interface that scripts will import.
    """

    if ode_name not in ODES:
        raise ValueError(f"Unknown ODE: {ode_name}. Choose from {list(ODES.keys())}")
    
    if method not in INTEGRATORS:
        raise ValueError(f"Unknown Mehtod: {method}. Choose from {list(INTEGRATORS.keys())}")
    
    ode_rhs = ODES[ode_name]
    x_eval = np.linspace(x_start, x_end, num_points)
    solution = solve_ivp(ode_rhs, t_span=(x_start, x_end), t_eval=x_eval, y0 = [u0], method=method)

    return solution

# CLI Tooling (Only runs when executed as main)

def get_args():
    """
        Isolates parser
    """
    parser = argparse.ArgumentParser(description="Generate and test ODE trajectories.")
    parser.add_argument(
        "--ode",
        choices = ODES.keys(),
        default = "bernoulli",
        help = "Name of the ODE to solve (default: bernoulli)"
    )

    #Extra arguments
    parser.add_argument("--start", type=float, default=1.0, help="Start x value")
    parser.add_argument("--end", type=float, default=10.0, help="End x value")
    parser.add_argument("--initial_condition", type=float, default=1.0, help="Initial condition u(start)")
    parser.add_argument("--num_points", type=int, default=30, help="Number of points per trajectory")
    parser.add_argument("--method", choices= INTEGRATORS.keys(), default="RK45", help="Numerical integrator choice, default RK45")

    return parser.parse_args()

def main():
    print("Running as main")

    #Extract arguments
    args = get_args()

    #Generate trajectory
    solution = generate_trajectory(
        ode_name=args.ode,
        x_start = args.start,
        x_end = args.end,
        u0 = args.initial_condition,
        num_points=args.num_points
    )

    print(f"Integrating {args.ode} ODE using {args.method}")
    print(f"Grid : ({args.start}, {args.end})")
    print(f"Initial condition: {args.initial_condition}")
    print(f"Number of points: {args.num_points}")

    if solution.success:
        plt.plot(solution.t, solution.y[0], label= f"{args.ode}_ode")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    else:
        print(f"Solver failed: {solution.message}")


if __name__ == "__main__":
    main()