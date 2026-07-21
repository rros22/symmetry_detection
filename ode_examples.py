"""
    - Five ODEs with point symmetries
    - Trajectory generation
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import debugging as db

# ODEs

def bernoulli_ode(x,u):
     """
        Bernoulli equation du/dx = 2*u/x - u^2*x^2
     """

     return 2*u/x - (u**2)*(x**2)

def rational_ode(x,u):
    """
        Rational equation with rotation symmetry du/dx = (u^3 + u*x^2 - u - x)/(x*u^2 + x^3 + u - x)
    """

    return (u**3 + u*x**2 - u - x)/(x*u**2 + x**3 + u - x)

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

def grad_bernoulli_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = 2*u/x - u^2*x^2
    """
    df_dx = -2 * u / (x**2) - 2 * (u**2) * x
    df_du = 2 / x - 2 * u * (x**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_rational_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = (u^3 + u*x^2 - u - x) / (x*u^2 + x^3 + u - x)
    """
    N = u**3 + u * (x**2) - u - x
    D = x * (u**2) + x**3 + u - x

    dN_dx = 2 * u * x - 1
    dN_du = 3 * (u**2) + x**2 - 1

    dD_dx = u**2 + 3 * (x**2) - 1
    dD_du = 2 * x * u + 1

    df_dx = (dN_dx * D - N * dD_dx) / (D**2)
    df_du = (dN_du * D - N * dD_du) / (D**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_riccati_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = x*u^2 - 2*u/x - 1/x^3
    """
    df_dx = u**2 + 2 * u / (x**2) + 3 / (x**4)
    df_du = 2 * x * u - 2 / x

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_scaling_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = u/x + x/u
    """
    df_dx = -u / (x**2) + 1 / u
    df_du = 1 / x - x / (u**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)


def grad_abel_ode(x, u):
    """
        Gradient [-df/dx, -df/du, 1] for du/dx = u^3 - x
    """
    df_dx = -1.0 * np.ones_like(x)
    df_du = 3 * (u**2)

    return np.stack([-df_dx, -df_du, np.ones_like(x)], axis=0)

# Examples and default parameters

ODES = {
    "bernoulli": bernoulli_ode,
    "rational": rational_ode,
    "riccati": riccati_ode,
    "scaling": scaling_ode,
    "abel": abel_ode
}

NORMALS = {
    "bernoulli": grad_bernoulli_ode,
    "rational": grad_rational_ode,
    "riccati": grad_riccati_ode,
    "scaling": grad_scaling_ode,
    "abel": grad_abel_ode
}

INTEGRATORS = {
    "RK45": "RK45",
    "RK23": "RK23",
    "DOP853": "DOP853",
    "Radau": "Radau",
    "BDF": "BDF",
    "LSODA": "LSODA",
}

ODE_DEFAULTS = {
    "rational": {
        "x_start": 0.75,
        "x_end": 2.0,
        "initial_conditions": np.linspace(1, 7, 20),
        "initial_condition": 1.0,
        "num_points": 50,
        "method": "RK45",
    },
    "bernoulli": {
        "x_start": 0.2,
        "x_end": 1.2,
        "initial_conditions": np.linspace(0.75, 2, 10),
        "initial_condition": 1.0,
        "num_points": 30,
        "method": "RK45",
    },
    "riccati": {
        "x_start": 1.0,
        "x_end": 5.0,
        "initial_conditions": np.linspace(0.5, 3, 10),
        "initial_condition": 1.0,
        "num_points": 40,
        "method": "RK45",
    },
    "scaling": {
        "x_start": 1.0,
        "x_end": 10.0,
        "initial_conditions": np.linspace(1, 5, 15),
        "initial_condition": 1.0,
        "num_points": 50,
        "method": "RK45",
    },
    "abel": {
        "x_start": 0.0,
        "x_end": 0.09,
        "initial_conditions": np.linspace(-2, 2, 200),
        "initial_condition": 0.0,
        "num_points": 200,
        "method": "RK45",
    },
}



# Generate a full trajectory
def generate_trajectory(ode_name, x_start=1, x_end=10, u0=1, num_points=30, method="RK45"):
    """
        Integrate one of the differential equations in the examples
    """

    if ode_name not in ODES:
        raise ValueError(f"Unknown ODE: {ode_name}. Choose from {list(ODES.keys())}")
    
    if method not in INTEGRATORS:
        raise ValueError(f"Unknown Mehtod: {method}. Choose from {list(INTEGRATORS.keys())}")
    
    ode_rhs = ODES[ode_name]
    x_eval = np.linspace(x_start, x_end, num_points)
    solution = solve_ivp(ode_rhs, t_span=(x_start, x_end), t_eval=x_eval, y0 = [u0], method=method)

    return solution

def generate_equation_manifold(ode_name, x_start=1, x_end=7, initial_conditions=np.linspace(0.75,2,10), num_points=30, method="RK45"):
    """ 
        o = len(initial_conditions): is the number of trajectories.
        n: is the dimension of the embedding (x, u, u'). In this example n = 3 since we are embbeding ODE trajectories into the first order Jet Space.
        t = num_points: is the number of datapoints per trajectory.

        1. Integrates one of the differential equations in the examples for "o" trajectories.
        2. Constructs a jet space embbeding of the trajectories, by appending the time, derivative wrt time coordinate to the state coordinate.
        3. Returns a np.array of dimensions (o,n,t)
    """

    # Input data
    db.print_configuration_rich(ode_name, x_start, x_end, initial_conditions, num_points, method)

    # Iterate over all initial conditions
    trajectories = []

    for ic in initial_conditions:
        solution = generate_trajectory(ode_name, x_start=x_start, x_end=x_end, u0=ic, num_points=num_points, method=method)
        embedding = np.array([solution.t, solution.y[0], ODES[ode_name](solution.t, solution.y[0])])
        trajectories.append(embedding)

    return np.array(trajectories)

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
    parser.add_argument("--start", type=float, default=None, help="Start x value")
    parser.add_argument("--end", type=float, default=None, help="End x value")
    parser.add_argument("--initial_condition", type=float, default=None, help="Initial condition u(start)")
    parser.add_argument("--initial_conditions", type=float,nargs="+", default=None, help="Initial conditions, e.g. --initial_conditions 1.0 1.1 1.2")
    parser.add_argument("--num_points", type=int, default=None, help="Number of points per trajectory")
    parser.add_argument("--method", choices= INTEGRATORS.keys(), default=None, help="Numerical integrator choice, default RK45")

    args = parser.parse_args()

    # Get the dictionary of default arguments for the requested ODE
    defaults = ODE_DEFAULTS.get(args.ode, {})

# Merge: Override default arguments with those from the CLI
    args.start = args.start if args.start is not None else defaults.get("x_start", 1.0)
    args.end = args.end if args.end is not None else defaults.get("x_end", 7.0)
    args.initial_condition = args.initial_condition if args.initial_condition is not None else defaults.get("initial_condition", 1.0)

    # Correct plural check and assignment:
    if args.initial_conditions is not None:
        args.initial_conditions = np.array(args.initial_conditions)
    else:
        args.initial_conditions = defaults.get("initial_conditions", np.linspace(0.75, 2, 10))
    
    args.num_points = args.num_points if args.num_points is not None else defaults.get("num_points", 30)
    args.method = args.method if args.method is not None else defaults.get("method", "RK45")

    return args

def main():
    print("Running as main")

    # Extract arguments
    args = get_args()

    # # Test 1: Generate trajectory
    # solution = generate_trajectory(
    #     ode_name=args.ode,
    #     x_start = args.start,
    #     x_end = args.end,
    #     u0 = args.initial_condition,
    #     num_points=args.num_points,
    #     method = args.method
    # )

    # print(f"Integrating {args.ode} ODE using {args.method}")
    # print(f"Grid : ({args.start}, {args.end})")
    # print(f"Initial condition: {args.initial_condition}")
    # print(f"Number of points: {args.num_points}")

    # if solution.success:
    #     plt.scatter(solution.t, solution.y[0], label= f"{args.ode}_ode")
    #     plt.xlabel("x")
    #     plt.ylabel("u")
    #     plt.show()

    # else:
    #     print(f"Solver failed: {solution.message}")
    
    # Test 2: Generate equation manifold
    X = generate_equation_manifold(
        ode_name=args.ode,
        x_start = args.start,
        x_end = args.end,
        initial_conditions= args.initial_conditions,
        num_points=args.num_points,
        method = args.method
    )
    
    fig = plt.figure(figsize=(6,6))
    ax0 = fig.add_subplot(131, projection='3d')
    ax1 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(133)

    ax2.set_box_aspect(1)
    ax0.set_aspect('equal')

    # Define grid of values of surface manifold over [x_min,x_max] x [u_min,u_max].
    x_min = np.min(X[:,0,:])
    x_max = np.max(X[:,0,:])
    u_min = np.min(X[:,1,:])
    u_max = np.max(X[:,1,:])

    # Define p axis limits. They are different for integrated trajectories and for the grid over which the surface is plotted.
    p_traj_min = np.min(X[:,2,:])
    p_traj_max = np.max(X[:,2,:])

    res = 20

    X_grid = np.linspace(x_min,x_max,res)
    U_grid = np.linspace(u_min,u_max,res)

    X_grid, U_grid = np.meshgrid(X_grid, U_grid)
    P_grid = ODES[args.ode](X_grid,U_grid)

    p_grid_min = np.min(P_grid)
    p_grid_max = np.max(P_grid)

    # Evaluate analytic normal field on the surface
    N = NORMALS[args.ode](X_grid,U_grid)
    fig, ax = db.scaled_3D_quiver(X_grid, U_grid, P_grid, N, (x_min, x_max), (u_min, u_max), (p_grid_min, p_grid_max))

    # Plotting
    ax0.plot_surface(X_grid,U_grid,P_grid, cmap='coolwarm')

    for trajectory in X:
        # ax0.plot(trajectory[0,:], trajectory[1,:], trajectory[2,:], color='k')
        ax1.plot(trajectory[0,:], trajectory[1,:], trajectory[2,:])
        ax2.plot(trajectory[0,:], trajectory[1,:])

    plt.show()

if __name__ == "__main__":
    main()