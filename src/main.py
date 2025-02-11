import numpy as np
import matplotlib.pyplot as plt
import casadi as ca


def dynamics(x, u, L):
    # Get paramaters of the system
    mass = L[0]
    f_0 = L[1]
    f_1 = L[2]
    f_2 = L[3]

    # Get statess iof the system
    v = x[0]

    # Driff dynamics
    f_x = (-1 / mass) * (f_0 + f_1 * v + f_2 * (v) ** 2)

    # Controlled driff
    g_x = 1 / mass

    # Dynamics of the system
    x_dot = f_x + g_x * u

    return x_dot


def f_rk4(x, u, ts, L):
    x = x
    u = u[0]

    k1 = dynamics(x, u, L)
    k2 = dynamics(x + (1 / 2) * ts * k1, u, L)
    k3 = dynamics(x + (1 / 2) * ts * k2, u, L)
    k4 = dynamics(x + ts * k3, u, L)
    # Compute forward Euler method
    x = x + (1 / 6) * ts * (k1 + 2 * k2 + 2 * k3 + k4)
    x = x[0,]
    return x


def plot_results(x, u, xd, t):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot x_estimate with label 'x_estimate'
    ax1.set_xticklabels([])
    (state_1,) = ax1.plot(t, x[0, 0 : t.shape[0]], color="#C04747", lw=1.5, ls="-")
    (state_1d,) = ax1.plot(t, xd[0, 0 : t.shape[0]], color="#00FF00", lw=1.5, ls="--")
    (state_3,) = ax2.plot(t, u[0, 0 : t.shape[0]], color="#478DC0", lw=1.5, ls="-")
    # Add a legend
    ax1.legend(
        [state_1, state_1d],
        [r"v", r"vd"],
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=3,
        handletextpad=0.1,
        borderaxespad=0.3,
        columnspacing=2,
    )

    ax2.legend(
        [state_3],
        [r"u"],
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=2,
        borderpad=0.5,
        labelspacing=0.5,
        handlelength=3,
        handletextpad=0.1,
        borderaxespad=0.3,
        columnspacing=2,
    )

    ax1.grid(color="#949494", linestyle="-.", linewidth=0.8)
    ax2.grid(color="#949494", linestyle="-.", linewidth=0.8)

    ax2.set_xlabel(r"Time", labelpad=5)
    ax2.set_ylabel(r"N", labelpad=5)
    ax1.set_ylabel(r"m/s", labelpad=5)
    # Show the plot
    plt.show()


def control_law(x, xd, L):
    # Get paramaters of the system
    mass = L[0]
    f_0 = L[1]
    f_1 = L[2]
    f_2 = L[3]

    # Compute error
    kp = 1
    error = kp * (x - xd)

    u_c = -mass * error + (f_1 * x + f_2 * (x) ** 2) + f_0

    return u_c


def main():

    # Define the time and smaple time for the simulation
    ts = 0.05
    final = 10
    t = np.arange(0, final + ts, ts, dtype=np.double)

    # Parameters of the system
    mass = 1
    f_0 = 0.1
    f_1 = 0.05
    f_2 = 0.005
    alpha = 3

    L = [mass, f_0, f_1, f_2]

    # Initial conditions of the system
    v_0 = 0.0
    x = np.zeros((1, t.shape[0] + 1), dtype=np.double)
    x[0, 0] = v_0

    # Empty control action
    u = np.zeros((1, t.shape[0]), dtype=np.double)

    # Desired Velocity
    xd = 2 * np.ones((1, t.shape[0] + 1), dtype=np.double)

    # Set up casadi
    u_op = ca.MX.sym("u_op", 1)
    p = ca.MX.sym("p", 2)

    # Cost Function
    Q = ca.DM.eye(1)
    f = (1 / 2) * u_op * Q * u_op

    # Split external parameters
    vd_opt = p[0]
    v_op = p[1]

    # Restrictions
    g_expr = (
        (v_op - vd_opt) * (-(1 / mass) * (f_0 + f_1 * v_op + f_2 * (v_op) ** 2))
        + (v_op - vd_opt) * (1 / mass) * u_op
        + alpha * (1 / 2) * (v_op - vd_opt) ** 2
    )

    # Boundaries of the optimization variables
    lbx = [-4.0]
    ubx = [4.0]

    qp = {"x": u_op, "p": p, "f": f, "g": g_expr}
    solver = ca.qpsol("solver", "qpoases", qp)

    # Simulation Loop
    for k in range(0, t.shape[0]):
        ## Compute control actions using casadi
        sol = solver(lbx=lbx, ubx=ubx, lbg=-ca.inf, ubg=0, p=[xd[:, k], x[:, k]])
        u_optimal = sol["x"].full()
        u[:, k] = u_optimal

        # Evolution of the system
        x[:, k + 1] = f_rk4(x[:, k], u[:, k], ts, L)

    # Plot results of the system
    plot_results(x, u, xd, t)


if __name__ == "__main__":
    main()
