
# diffusion_calculator.py
# -----------------------
# Author: Zeke (github.com/ztzkz) & Adam (2025)
#
# Zeke developed the original diffusion model and created this faster version using petsc4py.
# Adam made slight modifications to do large parameter sweeps for fitting against experimental
# data to determine physically realistic parameters.
# 
# Main simulation driver for the diffusion model.
# Handles argument parsing, PETSc solver setup, time integration, and saving results.
# Designed for batch or single-job execution with parameter sweep support.

from petsc4py import PETSc
import numpy as np
import argparse
import csv
import time
import os

def initial_conditions(u, ctx):
    """
    Set initial conditions for the simulation.

    Parameters:
        u (PETSc.Vec): The PETSc vector to initialize.
        ctx (dict): Context dictionary, may contain 'restart_data' for restarting from a previous state.
    """
    if ctx.get("restart_data") is not None:
        u.setArray(ctx["restart_data"])
    else:
        u.setValues(range(6), [0.99 , 2e-9, 2e-5 , 1e-10 , 1e-10, 1e-10 ])  # A, B, C, H_A, H_B, H_C
    u.assemble()

def rhs_function(ts, t, u, F, ctx):
    """
    Compute the right-hand side (RHS) of the ODE system for the diffusion model.

    Parameters:
        ts (PETSc.TS): PETSc time stepper object (unused).
        t (float): Current simulation time.
        u (PETSc.Vec): Current state vector.
        F (PETSc.Vec): Output vector for the RHS (to be filled in-place).
        ctx (dict): Context dictionary with model parameters and constants.

    Returns:
        int: 0 on success (required by PETSc interface).
    """
    A, B, C, HA, HB, HC = u.getArray(readonly=True)
    k = ctx["k"]
    N = ctx["N"]
    D = ctx["D"]
    sf = ctx["scale_factors"]
    SCALE_A = sf["SA"]
    SCALE_C = sf["SC"]

    NA_avail = max(N["A"] - A, 0)
    NB_avail = max(N["B"] - B, 0)
    NC_avail = max(N["C"] - C, 0)

    dA = (-k["AH"] * A + k["HA"] * HA * NA_avail / N["A"])
    dB = -k["BH"] * B + k["HB"] * HB * NB_avail / N["B"]
    dC = (-k["CH"] * C + k["HC"] * HC * NC_avail / N["C"])

    dHA = (k["AH"] * A - k["HA"] * HA * NA_avail / N["A"])  + D * (HB - HA) * 2 / SCALE_A / 0.5
    dHB = k["BH"] * B - k["HB"] * HB * NB_avail / N["B"] + D * (HA + HC - 2 * HB) / (0.5 - SCALE_A - SCALE_C) ** 2
    dHC = (k["CH"] * C - k["HC"] * HC * NC_avail / N["C"]) + D * (HB - HC) * 2 / SCALE_C / 0.5

    F.setValues(range(6), [dA, dB, dC, dHA, dHB, dHC])
    F.assemble()
    return 0

def monitor(ts, step, time, u, ctx, fol, job_id):
    """
    Monitor function called periodically during time integration.
    Saves intermediate results to .npz files.

    Parameters:
        ts (PETSc.TS): PETSc time stepper object.
        step (int): Current time step number.
        time (float): Current simulation time.
        u (PETSc.Vec): Current state vector.
        ctx (dict): Context dictionary with simulation state/history.
        fol (str): Output folder for saving .npz files.
        job_id (int): Job identifier for output file naming.
    """
    freq = 10000
    if step % freq == 0:
        print(f"Step {step:4d}: t = {time:.5f}")
        vals = u.getArray(readonly=True).copy()
        ctx["history"].append((time, vals))
        times = [x[0] for x in ctx["history"]]
        ut = [x[1] for x in ctx["history"]]
        np.savez(f"{fol}/solution_{job_id}.npz", times=times, ut=ut, timestep=ts.getTimeStep(), args=vars(ctx["args"]), reason=-999)

def main():
    """
    Main entrypoint for running a diffusion simulation.
    Parses command-line arguments, sets up PETSc, runs the solver, and saves results.

    Parameters:
        None (uses command-line arguments)

    Command-line Arguments:
        -id, --job_id (int): The id of the job.
        -fol, --sub_folder (str): Subfolder to save data to.
        -L, --sample_length (float): Sample length (um).
        -T, --temp (float): Temperature (K).
        -D, --diffusion_rate (float): Diffusion constant D0.
        -hop, --hopping (float): Interstitial hopping barrier (eV).
        --debug (str): Whether to print output and intermediate values or not.
        -ts, --timestep (float): Timestep of simulation(s).
        -MT, --max_time (float): Max time to simulate (s).
        -AH, --A_detrap (float): HA activation energy (eV).
        -HA, --A_trap (float): AH activation energy (eV).
        -AHF, --A_detrap_attemptfreq (float): HA attempt frequency.
        -HAF, --A_trap_attemptfreq (float): AH attempt frequency.
        -BH, --B_detrap (float): HB activation energy (eV).
        -HB, --B_trap (float): BH activation energy (eV).
        -BHF, --B_detrap_attemptfreq (float): HB attempt frequency.
        -HBF, --B_trap_attemptfreq (float): BH attempt frequency.
        -CH, --C_detrap (float): HC activation energy (eV).
        -HC, --C_trap (float): CH activation energy (eV).
        -CHF, --C_detrap_attemptfreq (float): HC attempt frequency.
        -HCF, --C_trap_attemptfreq (float): CH attempt frequency.
    """
    parser = argparse.ArgumentParser(description="Diffusion model parameters")
    parser.add_argument('-id','--job_id',type=int,default=1,help='The id of the job')
    parser.add_argument('-fol','--sub_folder',type=str,default='./',help='Subfolder to save data to')
    parser.add_argument('-L','--sample_length',type=float,default=300,help='Sample length (um)')
    parser.add_argument('-T','--temp', type=float, default=623, help='Temperature (K)')
    parser.add_argument('-D','--diffusion_rate', type=float, default=1e-2, help='Diffusion constant D0')
    parser.add_argument('-hop','--hopping', type=float, default=.5, help='Interstitial hopping barrier (eV)')
    parser.add_argument('--debug',type=str,default='False',help='Whether to print output and intermediate values or not.')
    parser.add_argument('-ts','--timestep', type=float, default=1e-8, help='Timestep of simulation(s)')
    parser.add_argument('-MT','--max_time', type=float, default=1500, help='Max time to stimulate(s)')
    parser.add_argument('-AH','--A_detrap', type=float, default=1.8, help='HA activation energy (eV)')
    parser.add_argument('-HA','--A_trap', type=float, default=0, help='AH activation energy (eV)')
    parser.add_argument('-AHF','--A_detrap_attemptfreq', type=float, default=1e12, help='HA attempt frequency')
    parser.add_argument('-HAF','--A_trap_attemptfreq', type=float, default=0, help='AH attempt frequency')
    parser.add_argument('-BH','--B_detrap', type=float, default=0.7, help='HB activation energy (eV)')
    parser.add_argument('-HB','--B_trap', type=float, default=0.5, help='BH activation energy (eV)')
    parser.add_argument('-BHF','--B_detrap_attemptfreq', type=float, default=1e12, help='HB attempt frequency')
    parser.add_argument('-HBF','--B_trap_attemptfreq', type=float, default=1e10, help='BH attempt frequency')
    parser.add_argument('-CH','--C_detrap', type=float, default=1.3, help='HC activation energy (eV)')
    parser.add_argument('-HC','--C_trap', type=float, default=0.5, help='CH activation energy (eV)')
    parser.add_argument('-CHF','--C_detrap_attemptfreq', type=float, default=1e12, help='HC attempt frequency')
    parser.add_argument('-HCF','--C_trap_attemptfreq', type=float, default=1e8, help='CH attempt frequency')

    args = parser.parse_args()
    start_time = time.time()

    kb = 8.617333262145e-5
    job_id = args.job_id
    fol = args.sub_folder
    os.makedirs(fol, exist_ok=True)

    debug = args.debug.lower() == "true"

    T = args.temp
    L = args.sample_length
    D = (args.diffusion_rate / (L * 1e-5) ** 2) * np.exp(-args.hopping / (kb * T))
    max_time = args.max_time
    timestep = args.timestep

    SCALE_A = 0.1 / L
    SCALE_C = 0.05 / L
    SCALE_B = 0.2 / 50

    restart_data = None
    history = []
    npz_path = os.path.join(fol, f"solution_{job_id}.npz")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            if 'ut' in data and len(data['ut']) > 0:
                restart_data = data['ut'][-1]
                history = list(zip(data['times'], data['ut']))
                print("Restarting from saved state with", len(history), "steps")
        except Exception as e:
            print("Failed to load existing .npz for restart:", e)

    u = PETSc.Vec().createSeq(6)
    F = PETSc.Vec().createSeq(6)

    ctx = {
        "k": {
            "AH": args.A_detrap_attemptfreq * np.exp(-args.A_detrap / (kb * T)),
            "HA": args.A_trap_attemptfreq * np.exp(-args.A_trap / (kb * T)),
            "BH": args.B_detrap_attemptfreq * np.exp(-args.B_detrap / (kb * T)),
            "HB": args.B_trap_attemptfreq * np.exp(-args.B_trap / (kb * T)),
            "CH": args.C_detrap_attemptfreq * np.exp(-args.C_detrap / (kb * T)),
            "HC": args.C_trap_attemptfreq * np.exp(-args.C_trap / (kb * T)),
        },
        "N": {"A": 1, "B": 2e-4, "C": 1e-3},
        "D": D,
        "L": L,
        "scale_factors": {"SA": SCALE_A, "SB": SCALE_B, "SC": SCALE_C},
        "args": args,
        "history": history,
        "restart_data": restart_data,
    }

    ts = PETSc.TS().create()
    ts.setType("rk")
    ts.setRKType('3bs')
    ts.setTolerances(rtol=1e-3, atol=1e-8)
    ts.setProblemType(PETSc.TS.ProblemType.LINEAR)
    ts.setRHSFunction(rhs_function, F, args=(ctx,))
    ts.setTimeStep(timestep)
    ts.setMaxTime(max_time)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)
    ts.setFromOptions()
    ts.setMonitor(monitor, args=(ctx, fol, job_id))

    initial_conditions(u, ctx)
    ts.setSolution(u)

    # Initial save before solving
    if len(history) == 0:
        np.savez(npz_path, times=[], ut=[], timestep=timestep, args=vars(args), reason=-999)

    ts.solve(u)

    times = [x[0] for x in ctx["history"]]
    ut = [x[1] for x in ctx["history"]]
    reason = ts.getConvergedReason()
    np.savez(npz_path, times=times, ut=ut, timestep=ts.getTimeStep(), args=vars(args), reason=reason)

    end_time = time.time()
    print(f"Run complete. Reason: {reason}. Runtime: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    main()
