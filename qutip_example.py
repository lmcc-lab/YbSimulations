from qutip import *
import numpy as np
import matplotlib.pyplot as plt

def motion_qubit():
    up_ket = fock(2, 0)
    down_ket = fock(2, 1)
    up_rho = fock_dm(2, 0)
    down_rho = fock_dm(2,1)
    harmonic_size = 30
    thermal_5 = thermal_dm(harmonic_size, 5)
    Delta = 0.
    motion_identity = identity(harmonic_size)
    rabi_rate = 2 * np.pi * 2.0
    omega_s = 5.0
    adag = create(harmonic_size)
    a = destroy(harmonic_size)
    ldp = 0.1
    qubit_identity = identity(2)
    H_int = 0.5 * rabi_rate * (tensor(motion_identity, sigmax()) - ldp * tensor(adag + a, sigmay()))
    H_0 = 0.5 * Delta * tensor(motion_identity, sigmaz()) + omega_s * tensor(adag * a, qubit_identity)

    t = np.linspace(0, 10, 1000)
    eops = [tensor(motion_identity, down_rho)]#, adag * a, sigmax(), sigmay(), sigmaz()]
    cops = [np.sqrt(1.) * tensor(motion_identity,sigmaz())]
    init_rho = tensor(thermal_5, up_rho)
    result = mesolve(H_0 + H_int, init_rho, t, c_ops=cops, e_ops=eops)

    plt.plot(t, result.expect[0])
    plt.show()

def easy_qubit():
    up_ket = fock(2, 0)
    down_ket = fock(2, 1)
    up_rho = fock_dm(2, 0)
    down_rho = fock_dm(2,1)
    harmonic_size = 30
    thermal_5 = thermal_dm(harmonic_size, 5)

    rabi_rate = 2 * np.pi * 2.0
    H_int = 0.5 * rabi_rate * (up_ket * down_ket.dag() + down_ket * up_ket.dag())
    H_0 = 0. * up_rho + 0. * down_rho

    t = np.linspace(0, 10, 1000)
    eops = [down_rho]
    result = mesolve(H_0 + H_int, up_rho, t, c_ops=[], e_ops=eops)

    plt.plot(t, result.expect[0])
    plt.show()

if __name__ == '__main__':
    motion_qubit()