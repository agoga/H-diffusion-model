from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

def rhs_function(ts, t, u, F, ctx):
    zero = ctx['vec_zero']
    Hint, Hd = u.getSubVector(ctx['is_Hint']), u.getSubVector(ctx['is_Hd'])
    F_Hint, F_Hd = F.getSubVector(ctx['is_Hint']), F.getSubVector(ctx['is_Hd'])
    ktrap, kdetrap, Nd = ctx['ktrap'], ctx['kdetrap'], ctx['Nd']
    Nd_avail_frac = Nd.duplicate()
    Nd_avail_frac.pointwiseMult(zero, zero)     # zero out avail
    Nd_avail_frac.waxpy(-1.0, Hd, Nd)          # avail = Ndb - Hdb
    Nd_avail_frac.pointwiseMax(Nd_avail_frac, zero)     # avail = max(Ndb - Hdb, 0)
    Nd_avail_frac.pointwiseDivide(Nd_avail_frac, Nd) 
    F_Hint.setValues(range(ctx['m']), -ktrap * Nd_avail_frac * Hint + kdetrap * Hd)
    F_Hd.setValues(range(ctx['m']), ktrap * Nd_avail_frac * Hint - kdetrap * Hd)
    F.restoreSubVector(ctx['is_Hint'], F_Hint)
    F.restoreSubVector(ctx['is_Hd'], F_Hd)
    u.restoreSubVector(ctx['is_Hint'], Hint)
    u.restoreSubVector(ctx['is_Hd'], Hd)
    return 0


def ifunction(ts, t, u, udot, F, ctx):
    A_Hint = ctx['A_Hint']          # Laplacian matrix
    D = ctx['D']

    Hint = u.getSubVector(ctx['is_Hint'])
    Hd = u.getSubVector(ctx['is_Hd'])
    dHint = udot.getSubVector(ctx['is_Hint'])
    dHdb = udot.getSubVector(ctx['is_Hd'])
    F_Hint = F.getSubVector(ctx['is_Hint'])
    F_Hd = F.getSubVector(ctx['is_Hd'])

    # Compute F_Hint = dHint - D * A * Hint
    A_Hint.mult(Hint, F_Hint)
    F_Hint.scale(-D)
    F_Hint.axpy(1.0, dHint)

    # For Hd: F = du/dt (pure ODE)
    F_Hd.set(0.0)
    F_Hd.axpy(1.0, dHdb)

    # Restore
    F.restoreSubVector(ctx['is_Hint'], F_Hint)
    F.restoreSubVector(ctx['is_Hd'], F_Hd)
    u.restoreSubVector(ctx['is_Hint'], Hint)
    u.restoreSubVector(ctx['is_Hd'], Hd)
    udot.restoreSubVector(ctx['is_Hint'], dHint)
    udot.restoreSubVector(ctx['is_Hd'], dHdb)

    return 0


def ijacobian(ts, t, u, udot, shift, J, P, ctx):
    A, D = ctx['A'], ctx['D']
    J.zeroEntries()
    A.copy(result=J)
    J.scale(-D)
    J.shift(shift)  # J = shift*I - D*A
    J.assemble()
    if P != J:
        P.assemble()
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

def assemble_laplacian(A, m, x):
    A.zeroEntries()
    for i in range(1, m - 1):
        h_left = x[i] - x[i - 1]
        h_right = x[i + 1] - x[i]
        A.setValue(i, i - 1,  2.0 / (h_left * (h_left + h_right)))
        A.setValue(i, i,    -2.0 / (h_left * h_right))
        A.setValue(i, i + 1, 2.0 / (h_right * (h_left + h_right)))

    # Neumann BC at left boundary (approximate d²u/dx² using ghost symmetry)
    hL = x[1] - x[0]
    A.setValue(0, 0, -2.0 / hL**2)
    A.setValue(0, 1,  2.0 / hL**2)

    # Neumann BC at right boundary
    hR = x[-1] - x[-2]
    A.setValue(m-1, m-2, 2.0/hR**2)
    A.setValue(m-1, m-1, -2.0/hR**2)

    A.assemble()

def monitor(ts, step, time, u, ctx):
    x = ctx['x']
    L = ctx['L']

    if step % 500 == 0:
        Hint, Hd = u.getSubVector(ctx['is_Hint']).getArray(readonly=True), u.getSubVector(ctx['is_Hd']).getArray(readonly=True)
        ctx['Hint_history'].append(Hint.copy())
        ctx['Hd_history'].append(Hd.copy())
        ctx['t_history'].append(time)
        print(f"Step {step:4d}: t = {time:.5f}")
        plt.figure()
        plt.plot(x[0:-4]*L, Hint[0:-4]*1e22, label='H (mobile)')
        plt.plot(x[0:-4]*L, Hd[0:-4]*1e22, label='H (trapped)')
        plt.title(f"Solution at step {step:4d}, t={time:.4f}")
        plt.xlabel('x(nm)')
        plt.ylabel('H(cm^-3)')
        plt.yscale('log')
        plt.ylim(1e11, 1e22)
        plt.legend()
        plt.savefig(f"sample_{step:06d}.png")
        np.savez(f"sample_{step:06d}.npz", Hint=Hint, Hd=Hd, t=time)
        plt.close()

def initial_conditions(u, ctx):
    m = ctx['m']
    Hint, Hd = u.getSubVector(ctx['is_Hint']), u.getSubVector(ctx['is_Hd'])
    Hint_arr = Hint.getArray()
    Hd_arr = Hd.getArray()
    for i in range(m):
        Hint_arr[i] = 1e-8
        Hd_arr[i] = 1e-8
    # Hint_arr[0:2] = 1e-5
    Hd_arr[0:5] = 0.99
    # Hd_arr[-1] = 0
    # Hd_arr[-5:] = 5e-5
    Hint.setArray(Hint_arr)
    Hd.setArray(Hd_arr)
    u.restoreSubVector(ctx['is_Hint'], Hint)
    u.restoreSubVector(ctx['is_Hd'], Hd)

def main():
    SAMPLE_LENGTH = 1000 #nm
    print(PETSc.ScalarType)
    T = 950
    kb = 8.617333262145e-5  # Boltzmann constant in eV/K
    # Scaled Diffusion coefficient in sm^2/s where sample length sm = 1e-4 m
    D = 1e-3 / (SAMPLE_LENGTH*1e-8)**2 * np.exp(-0.5 / (kb * T))  
    x = np.array((np.linspace(0,8,5)/SAMPLE_LENGTH).tolist()+
                 (np.linspace(10,300, 10)/SAMPLE_LENGTH).tolist() +
                [300.5/SAMPLE_LENGTH, 301/SAMPLE_LENGTH, 301.5/SAMPLE_LENGTH] +
                (np.linspace(302, SAMPLE_LENGTH, 8)/SAMPLE_LENGTH).tolist() +
                [50,100])
    m = 28
    ktrap = np.ones_like(x) * 1e10 * np.exp(-0.5 / (kb * T))
    ktrap[0:5] = 1e12 * np.exp(-0.8 / (kb * T))
    ktrap[15:18] = 1e10 * np.exp(-1.2 / (kb * T))
    # ktrap[-5:] = 1e8 * np.exp(-0.5 / (kb * T))
    # ktrap[-1] = 0  #Ghost point
    kdetrap = np.ones_like(x) * 1e12 * np.exp(-0.7 / (kb * T))
    kdetrap[0:5] = 1e12 * np.exp(-2.2 / (kb * T))
    kdetrap[15:18] = 1e12 * np.exp(-2.3 / (kb * T))
    # kdetrap[-5:] = 1e12 * np.exp(-1.3 / (kb * T))
    # kdetrap[-1] = 0  #Ghost point
    Nd = np.ones_like(x) * 2e-4
    Nd[0:5] = 1
    Nd[15:18] = 1e-2
    Nd[18:] = 1e-6
    # Nd[-1] = 0  #Ghost point
    ktrap_vec = PETSc.Vec().createSeq(m)
    ktrap_vec.setArray(ktrap)
    kdetrap_vec = PETSc.Vec().createSeq(m)
    kdetrap_vec.setArray(kdetrap)
    Nd_vec = PETSc.Vec().createSeq(m)
    Nd_vec.setArray(Nd)

    ctx = {
        'x': x,
        'm': m,
        'D': D,
        'ktrap': ktrap_vec,
        'kdetrap': kdetrap_vec,
        'Nd': Nd_vec,
        'L' : SAMPLE_LENGTH
    }
    ctx['vec_zero'] = PETSc.Vec().createSeq(m)
    ctx['vec_zero'].set(0.0)

    is_Hint = PETSc.IS().createStride(m, 0, 1, comm=PETSc.COMM_SELF)
    is_Hd = PETSc.IS().createStride(m, m, 1, comm=PETSc.COMM_SELF)
    ctx['is_Hint'] = is_Hint
    ctx['is_Hd'] = is_Hd

    A = PETSc.Mat().createAIJ([2 * m, 2 * m], nnz=4)
    A.setUp()
    assemble_laplacian(A, m, x)
    ctx['A'] = A

    A_Hint = PETSc.Mat().createAIJ([m, m], nnz=4)
    A_Hint.setUp()
    assemble_laplacian(A_Hint, m, x)
    ctx['A_Hint'] = A_Hint

    u = PETSc.Vec().createSeq(2 * m)
    J = PETSc.Mat().createAIJ([2 * m, 2 * m], nnz=4)
    J.setUp()
    J.assemble()

    ts = PETSc.TS().create()
    ts.setType("arkimex")
    ts.setARKIMEXType("prssp2") 
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)

    ts.setIFunction(ifunction, None, args=(ctx,))
    ts.setIJacobian(ijacobian, J, J, args=(ctx,))
    ts.setRHSFunction(rhs_function, None, args=(ctx,))
    ts.setTimeStep(1e-8) 
    ts.setMaxTime(0.01)
    # ts.setMaxSteps(20000000)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)
    
    ts.setTolerances(rtol=1e-3, atol=1e-7)

    ctx['Hint_history'] = []
    ctx['Hd_history'] = []
    ctx['t_history'] = []
    ts.setMonitor(monitor, args=(ctx,))

    initial_conditions(u, ctx)
    RESTART = False
    if RESTART:
        # Load saved state
        data = np.load("/home/zeke/Desktop/petsc/TOPCon/solution0_81600000.npz")
        Hint_array = data["Hint"]
        Hd_array   = data["Hd"]
        t0         = float(data["t"])

        # Apply to u
        Hint_vec = u.getSubVector(ctx['is_Hint'])
        Hd_vec   = u.getSubVector(ctx['is_Hd'])
        Hint_vec.setArray(Hint_array)
        Hd_vec.setArray(Hd_array)
        u.restoreSubVector(ctx['is_Hint'], Hint_vec)
        u.restoreSubVector(ctx['is_Hd'], Hd_vec)

        # Set TS time and continue
        ts.setTime(t0)

    ts.setSolution(u)
    ts.setFromOptions()
    ts.solve(u)

if __name__ == "__main__":
    main()
