#!/usr/bin/env python3

import PyDSTool as dst
from PyDSTool.Toolbox import phaseplane as pp
import matplotlib.pyplot as plt
import math


def initialise_fast_subsystem():
    """
    Returns a PyDSTool args object for the Hindmarsh Rose fast
    subsystem.
    """
    DS_args = dst.args(name="HR_fast")
    DS_args.pars = {"a": 1, "b": 3, "c": 1, "d": 5, "z": 0, "I": 2}
    DS_args.varspecs = {"x": "y - a*x^3 + b*x^2 - z + I", "y": "c - d*x^2 - y"}
    DS_args.fnspecs = {
        "Jacobian": (["t", "x", "y"], "[[-a*3*x^2 + 2*b*x, 1],[-2*d*x, -1]]")
    }

    DS_args.tdomain = [0, 30]
    DS_args.xdomain = {"x": [-2, 3], "y": [-10, 0]}
    DS_args.ics = {"x": 0, "y": -2}

    return DS_args


def initialise_full_system():
    """
    Return a PyDSTool object for the full Hindmarsh Rose system,
    initialised at a useful intial condition
    """
    HRFull = dst.args(name="HindmarshRoseFull")  # Name of the system
    HRFull.pars = {  # Parameter names and values
        "a": 1,
        "b": 3,
        "c": 1,
        "d": 5,
        "r": 1e-3,
        "s": 4,
        "xr": -1.6,
        "I": 2,
    }
    HRFull.varspecs = {  # ODE definition
        "x": "y - a*x^3 + b*x^2 -z + I",
        "y": "c - d*x^2 - y",
        "z": "r*(s*(x-xr) - z)",
    }

    # Initial conditions taken from PyDSTool Hindmarsh Rose example.
    # Avoids having to wait for transients to decay
    x0 = -0.5 * (1 + math.sqrt(5))
    HRFull.ics = {
        "x": x0,
        "y": 1 - 5 * x0 ** 2,
        "z": HRFull.pars["I"],
    }
    HRFull.tdomain = [0, 500]

    return HRFull


def integrate_and_plot(
    x_var,
    y_var,
    DS_args,
    x_label=None,
    y_label=None,
    title=None,
    plot=None,
    label=None,
    sample_range=(None, None),
    show_IC=False,
    linetype="",
):
    """
    Take a string representing the dependent variable, and a PyDSTool
    args object representing a dynamical system. Perform a time
    integration on the system, and plot the dependent variable against
    time.

    kwargs:
        x_label: optional x label for the plot (default dependent_var)
        y_label: optional y label for the plot (default 't')
        title: optional plot title (default nothing)
        plot: tuple (fig,ax) on which to plot the data; generates a new
              fig,ax pair if not specified. fig can be None.
        label: legend label for the plotted trajectory, default None
        sample_range = 2-tuple of form (lower bound, upper bound),
                       giving start and end times to sample the
                       trajectory at
        show_IC: show the initial condition on the plot
        linetype: optional line formatter to pass to matplotlib

    Side effects:
        Modifies the passed ax object, if any, to include data plots
    """

    t_low, t_high = sample_range
    ODE = dst.Generator.Vode_ODEsystem(DS_args)
    trajectory = ODE.compute("test_traj").sample(tlo=t_low, thi=t_high)

    fig, ax = plot if plot else plt.subplots()
    ax.plot(trajectory[x_var], trajectory[y_var], linetype, label=label)
    ax.set_xlabel(x_label if x_label else x_var)
    ax.set_ylabel(y_label if y_label else y_var)
    if show_IC:
        ax.scatter(DS_args.ics[x_var], DS_args.ics[y_var], label="IC")
    if title:
        ax.set_title(title)
    return fig, ax, trajectory


def build_phase_plane(DS_args, axes=None):
    """
    Taken from the PyDSTool phaseplane toolbox examples. Not a very
    reusable piece of code, in its current form, but it illustrates
    how to use the software.
    If axes=(fig,ax), the phase plane will be plotted on the provided
    axis. If axes=None, a new plot will be created. The plot objects
    are returned.
    """
    fig, ax = axes if axes else plt.subplots()
    ODE_sys = dst.Generator.Vode_ODEsystem(DS_args)

    # Plot the vector field
    pp.plot_PP_vf(ODE_sys, "x", "y")

    # Find nullclines. n=3 uses three starting points in the domain to
    # find nullcline parts, to an accuracy of eps=1e-8, and a maximum
    # step for the solver of 0.1 units. The fixed point found is also
    # provided to help locate the nullclines.
    nulls_x, nulls_y = pp.find_nullclines(
        ODE_sys, "x", "y", n=3, eps=1e-8, max_step=0.1
    )

    # plot the nullclines
    ax.plot(nulls_x[:, 0], nulls_x[:, 1], "b", label="x nullcline")
    ax.plot(nulls_y[:, 0], nulls_y[:, 1], "g", label="y nullcline")

    # Only one fixed point, hence the [0] at end to extract it. n=4
    # uses four starting points in the domain to find the fixed point,
    # to an accuracy of eps=1e-8.
    fp_coord = pp.find_fixedpoints(ODE_sys, n=4, eps=1e-8)[0]
    ax.scatter(fp_coord["x"], fp_coord["y"], label="fixed point", s=100)
    print("Fixed point at {0}".format(fp_coord))

    return fig, ax


def find_fixedpoints(DS_args, n=4, eps=1e-8):
    """
    Find fixed points of the system defined by DS_args, taking
    n initial points, and a target accuracy of eps. Return a
    list of fixedpoint_2D objects.
    """
    ODE_sys = dst.Generator.Vode_ODEsystem(DS_args)
    fp_coords = pp.find_fixedpoints(ODE_sys, n=n, eps=eps)
    fps = [
        pp.fixedpoint_2D(ODE_sys, dst.Point(fp_coord), eps=eps)
        for fp_coord in fp_coords
    ]
    return fps


def continue_equilibrium(
    DS_args,
    equilibrium,
    free_pars,
    continuation_name="EPC",
    verbose=True,
    bifurcations=["LP", "H"],
    args_pycont=None,
):
    """
    Run numerical continuation on an equilibrium of a system.

        DS_args : the system to run the continuation on
        equilibrium : where to start the continuation from
        free_pars : which parameter to continue in
        continuation_name : a name for the resulting continuation
            object
        verbose : whether to display lots of information
        bifurcations : list of bifurcations to apply test functions
            for
        args_pycont : optional initialised continuation arguments
            object
    Returns a pycont continuation object.
    """
    # Detect singletons
    if isinstance(free_pars, str):
        free_pars = [free_pars]

    # Construct a simulatable ODE, with initial conditions at the
    # equilibrium
    ODE = dst.Generator.Vode_ODEsystem(DS_args)
    ODE.set(ics=equilibrium)

    # Initialise the continuation object, if not already done
    if args_pycont:
        pycont_args = args_pycont
    else:
        pycont_args = dst.args(name=continuation_name, type="EP-C")
        pycont_args.freepars = free_pars
        pycont_args.LocBifPoints = bifurcations
        pycont_args.SaveEigen = True
        pycont_args.SaveJacobian = True
        pycont_args.MaxNumPoints = 400
        if verbose:
            pycont_args.verbosity = 2

    # Continue!
    pycont = dst.ContClass(ODE)
    pycont.newCurve(pycont_args)
    pycont[continuation_name].forward()
    pycont[continuation_name].backward()
    if verbose:
        pycont[continuation_name].info()

    return pycont


def track_LC_from_pycont_Hopf(
    pycont, curve_name, point_name, free_var, pycont_args=None
):
    """
    Given a Hopf bifurcation, continue the limit cycles emerging from
    it.
        pycont : a continuation object, as returned from an
            equilibrium continuation
        curve_name : a name for the family of continued limit cycles
        point_name : the label of the Hopf bifurcation to start the
            continuation at
        free_var : the free parameter we're continuing in
        pycont_args : optional continuation arguments object
    """
    if pycont_args is None:
        pycont_args = dst.args(name=curve_name, type="LC-C")
        pycont_args.freepars = [free_var]
        pycont_args.SaveEigen = True
        pycont_args.verbosity = 2
        pycont_args.initpoint = point_name
    pycont.newCurve(pycont_args)
    pycont[curve_name].forward()
    pycont[curve_name].backward()
    return pycont


def main():
    # Get an args object representing the fast subsystem
    fast_subsystem_args = initialise_fast_subsystem()

    # Plot a simulated trajectory
    fig, ax = plt.subplots()
    integrate_and_plot(
        "t",
        "x",
        fast_subsystem_args,
        title="Fast subsystem dynamics",
        plot=(fig, ax),
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    # Build proper phase plane, and find fixed points
    fig, ax = plt.subplots()
    build_phase_plane(fast_subsystem_args, (fig, ax))
    # Overlay a LC trajectory
    integrate_and_plot(
        "x",
        "y",
        fast_subsystem_args,
        plot=(fig, ax),
        label="Limit cycle",
        sample_range=(20, 30),
        linetype="r",
    )
    ax.set_title("Phase plane")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    # Find equilibria
    fps = find_fixedpoints(fast_subsystem_args)
    equilibrium = fps[0].point
    # Numerically continue equilibrium in z
    pycont = continue_equilibrium(fast_subsystem_args, equilibrium, "z")
    # Plot, display, save
    plt.figure()
    pycont["EPC"].display(("z", "x"), stability=True)
    plt.xlim(-11,4)
    plt.title("Equilibrium continuation")
    plt.tight_layout()
    plt.show()
    plt.close()

    # track limit cycles
    plt.figure()
    # Numerically continue LCs from first Hopf
    track_LC_from_pycont_Hopf(pycont, "HC1", "EPC:H1", "z")
    pycont["HC1"].display(coords=("z", "x_max"), stability=True)
    pycont["HC1"].display(coords=("z", "x_min"), stability=True)
    # Numerically continue LCs from second Hopf
    track_LC_from_pycont_Hopf(pycont, "HC2", "EPC:H2", "z")
    pycont["HC2"].display(coords=("z", "x_max"), stability=True)
    pycont["HC2"].display(coords=("z", "x_min"), stability=True)
    # Format plot, display and save
    pycont["EPC"].display(("z", "x"), stability=True)
    plt.title("Limit cycle continuation")
    plt.xlim(-11, 4)
    plt.tight_layout()
    plt.show()
    plt.close()

    # overlay burster trajectory
    # Get args object representing full HR system
    hr_full = initialise_full_system()
    # Plot the same bifurcation diagram as before
    pycont["HC1"].display(coords=("z", "x_max"), stability=True)
    pycont["HC1"].display(coords=("z", "x_min"), stability=True)
    pycont["EPC"].display(("z", "x"), stability=True)
    # Simulate the full system
    traj = dst.Generator.Vode_ODEsystem(hr_full).compute('').sample()
    # Plot its trajectory on the bifurcation diagram
    plt.plot(traj['z'], traj['x'])
    # Tidy up plot and save
    plt.title("Bursting dynamics")
    plt.xlim(1.5, 3.1)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
