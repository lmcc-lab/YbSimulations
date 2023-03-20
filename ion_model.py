import numpy as np
from numpy import pi as pi
from scipy.constants import h, c, hbar
from scipy.constants import physical_constants
from typing import Union
import matplotlib.pyplot as plt


class General:
    def __init__(self):
        self.gamma_2S12_2P12 = 2 * pi * 19.6e6    # radians/s
        self.lambda_2S12_2P12 = 369.5e-9          # m
        self.lambda_2D32_3D3212 = 935.2e-9        # m
        self.gamma_2D32_3D3212 = 2 * pi * 4.2e6   # radians / s
        self.mu_b = physical_constants['Bohr magneton'][0] / hbar  # kg m^2 s^-2
        self.zeeman_P_state_proportion = 1/3
        self.zeeman_S_state_proportion = 1

        self.branch_ratio_3D3212_2S12 = 0.982
        self.branch_ratio_2P12_2D32 = 1 - self.branch_ratio_3D3212_2S12
        self.hyperfine_splitting_2S12 = 2 * pi * 12.643e9  # radians/s
        self.hyperfine_splitting_2P12 = 2 * pi * 2.105e9   # radians/s
        self.hyperfine_splitting_2D32 = 2 * pi * 0.86e9    # radians/s
        self.hyperfine_splitting_3D32 = 2 * pi * 2.21e9    # radians/s

        self.optimal_thetaBE = np.arccos(1/np.sqrt(3))

        self.I370sat = pi * h * c * self.gamma_2S12_2P12 / ( 3 * self.lambda_2S12_2P12 ** 3)  # W/m^2
        self.I935sat = pi * h * c * self.gamma_2D32_3D3212 / (3 * self.lambda_2D32_3D3212 ** 3)  # W/m^2

        self.excited_lifetime_2P12 = 8.1e-9  #s
        self.excited_lifetime_3D32 = 37.7e-9  #s
        self.excited_lifetime_2D32 = 52.7e-3  #s

        self.Gamma_2P12 = 1/self.excited_lifetime_2P12
        self.Gamma_3D32 = 1/self.excited_lifetime_3D32
        self.Gamma_2D32 = 1/self.excited_lifetime_2D32

    def zeeman_shift(self, B):
        return self.mu_b * B

    def s0(self, I, Isat):
        return I / Isat

    def rabi_freq(self, s0, gamma):
        return np.sqrt(s0 / 2) * gamma

    def effective_linewidth(self, thetaBE, rabi_freq, zeeman_shift, gamma):
        return NotImplementedError

    def excited_population_no_leakage(self, rabi_freq, effective_linewidth, thetaBE, detuning):
        return NotImplementedError

    def excited_population_with_leakage(self, excited_pop_no_leak, eta):
        return excited_pop_no_leak / (1 + eta * excited_pop_no_leak)

    def eta(self, excited_pop_D32, epsilon=1e-8):
        return self.branch_ratio_2P12_2D32 * self.gamma_2S12_2P12 / \
               (self.branch_ratio_3D3212_2S12 * self.gamma_2D32_3D3212) * 1 / (excited_pop_D32 + epsilon)





class Yb171(General):

    def effective_linewidth(self, thetaBE, rabi_freq, zeeman_shift, gamma):
        half_linewidth_squared = (gamma/2)**2
        cos2_thetaBE = np.cos(thetaBE)**2
        return half_linewidth_squared + \
               rabi_freq**2/3 * cos2_thetaBE * (1 - 3 * cos2_thetaBE)/(1 + 3 * cos2_thetaBE) + \
               cos2_thetaBE / (1 + 3 * cos2_thetaBE) * (rabi_freq ** 4 / (36 * zeeman_shift ** 2) + 4 * zeeman_shift ** 2)

    def excited_population_no_leakage(self, rabi_freq, effective_linewidth, thetaBE, detuning):
        cos2_thetaBE = np.cos(thetaBE) ** 2
        sin2_thetaBE = np.sin(thetaBE) ** 2
        return 3/4 * cos2_thetaBE * sin2_thetaBE / (1 + 3 * cos2_thetaBE) * (rabi_freq ** 2 / 3) / \
               (detuning**2 + effective_linewidth)


class Yb174(General):

    def effective_linewidth(self, thetaBE, rabi_freq, zeeman_shift, gamma):
        cos2 = np.cos(thetaBE)**2
        gamma92 = 9 * gamma ** 2
        zeeman162 = 16 * zeeman_shift ** 2 / gamma92
        return rabi_freq**2/6 + gamma**2/4 * (1 + zeeman162) *\
               (1 + 64 * zeeman_shift**2 / gamma92) / (1 + zeeman162 * 3 * cos2 + 1)

    def excited_population_no_leakage(self, rabi_freq, effective_linewidth, detuning):
        return 1/2 * (rabi_freq**2 / 6) / (detuning ** 2 + effective_linewidth)

    def eta(self):
        return 0.1


class PolarVectorGeneral:
    def __init__(self,
                 r: Union[float, np.array],
                 theta: Union[float, np.array],
                 phi: Union[float, np.array]):
        self.r = r
        self.theta = theta
        self.phi = phi

    def update(self, r, theta, phi):
        self.r = r
        self.theta = theta
        self.phi = phi

    def rotate_theta_by(self, radians):
        self.theta += radians

    def rotate_phi_by(self, radians):
        self.phi += radians

    def change_mag(self, change):
        self.r += r

    def convert_to_cartesian(self):
        x = self.r * np.sin(self.theta) * np.cos(self.phi)
        y = self.r * np.sin(self.theta) * np.sin(self.phi)
        z = self.r * np.cos(self.theta)
        return np.array([x, y, z])


def unwrap_angle(angles):
    """ Assumed that the angle should be going in one direction at all times """
    direction = np.sign(angles[1:] - angles[:-1])
    direction = np.abs((direction[1:] - direction[:-1]))
    change_index = np.argwhere(direction == 2)[:, 0]
    even = len(change_index) % 2 == 0

    if len(change_index) == 0:
        return angles
    skip = 1
    for i, ind in enumerate(change_index):
        if i < len(change_index) - 1 and skip:
            angles[ind: change_index[i+1] + 1] = angles[ind] + (pi - angles[ind: change_index[i+1] + 1])
            skip = 0
        elif not even and not skip:
            angles[ind:] = angles[ind] + (pi - angles[ind:])
        elif not skip:
            skip = 1

    return angles




class PolarVector(PolarVectorGeneral):

    def calculate_angle_between(self, other_vector: PolarVectorGeneral):
        vectors = [self, other_vector]
        r_variables = [self.r, other_vector.r]
        theta_variables = [self.theta, other_vector.theta]
        phi_variables = [self.phi, other_vector.phi]

        check_r = [type(xx) in [list, np.ndarray] for xx in r_variables]
        check_theta = [type(xx) in [list, np.ndarray] for xx in theta_variables]
        check_phi = [type(xx) in [list, np.ndarray] for xx in phi_variables]

        if not len(set(check_r))==1:
            to_change = check_r.index(False)
            vectors[to_change].r = np.zeros(len(r_variables[1 - to_change])) + r_variables[to_change]

        if not len(set(check_theta)) == 1:
            to_change = check_theta.index(False)
            vectors[to_change].theta = np.zeros(len(theta_variables[1 - to_change])) + theta_variables[to_change]

        if not len(set(check_phi)) == 1:
            to_change = check_phi.index(False)
            vectors[to_change].phi = np.zeros(len(phi_variables[1 - to_change])) + phi_variables[to_change]

        cart1 = vectors[0].convert_to_cartesian()
        cart2 = vectors[1].convert_to_cartesian()
        size = cart1.shape
        dot = np.zeros(size[1])
        for t in range(size[1]):
            curdot = np.vdot(cart1[:, t], cart2[:, t])
            dot[t] = curdot
        # check for wrapping
        angle = np.arccos(dot/(self.r * other_vector.r))
        unwrap_angle(angle)
        return angle, (cart1, cart2)


class GenMeshgrid:

    def __init__(self, poss_arrays: tuple, names: tuple):
        self.poss_arrays = poss_arrays
        self.names = names
        for i, name in enumerate(self.names):
            self.__setattr__(name, self.poss_arrays[i])
        self.variables_forming_meshgrid = (var for var in poss_arrays if type(var) == np.ndarray)
        self.mesh_order = {name: len(poss_arrays[i]) for i, name in enumerate(names) if type(poss_arrays[i]) == np.ndarray}
        self.variables_names = (names[i] for i, var in enumerate(poss_arrays) if type(var) == np.ndarray)
        self.mesh_shape = tuple(self.mesh_order.values())

    def gen_meshgrid(self):
        meshgrids = np.meshgrid(*self.variables_forming_meshgrid)
        for i, name in enumerate(self.variables_names):
            self.__setattr__(name, meshgrids[i])


def calculate_171_pop(detuning: Union[np.array, float],
                      I935: Union[np.array, float],
                      I370: Union[np.array, float],
                      thetaBE=None,
                      b_field=None,
                      e_field=None,
                      b_mag=None,
                      zeeman=None,
                      s=None):

    cart = None

    yb171 = Yb171()

    if thetaBE is not None and b_mag is not None:
        pass
    elif e_field is not None and b_field is not None:
        thetaBE, cart = b_field.calculate_angle_between(e_field)
        b_mag = b_field.r
    elif thetaBE is not None and zeeman is not None:
        pass
    else:
        raise ValueError("ThetaBE cannot be calculated. e_field, b_field or thetaBE, b_mag aren't supplied. "
                         "Either thetaBE and b_mag needs to be supplied or e_field and b_field.")

    if I370 is not None:
        s0_370 = yb171.s0(I370, yb171.I370sat)
    elif s is not None:
        s0_370 = s
    else:
        raise ValueError("I370 or s need to be passed")

    zeeman = yb171.zeeman_shift(b_mag) if zeeman is None else zeeman

    variables = (thetaBE, detuning, s0_370, I935, zeeman)

    mesh = GenMeshgrid(variables, ("thetaBE", "detuning", "s0", "I935", "zeeman"))
    mesh.gen_meshgrid()

    rabi_370 = yb171.rabi_freq(mesh.s0, yb171.gamma_2S12_2P12)
    eff_linewidth_370 = yb171.effective_linewidth(mesh.thetaBE, rabi_370, mesh.zeeman, yb171.gamma_2S12_2P12)
    excited_pop_370 = yb171.excited_population_no_leakage(rabi_370, eff_linewidth_370, mesh.thetaBE, mesh.detuning)

    s0_935 = yb171.s0(mesh.I935, yb171.I935sat)
    rabi_935 = yb171.rabi_freq(s0_935, yb171.gamma_2S12_2P12)
    eff_linewidth_935 = yb171.effective_linewidth(mesh.thetaBE, rabi_935, mesh.zeeman, yb171.gamma_2S12_2P12)
    excited_pop_935 = yb171.excited_population_no_leakage(rabi_935, eff_linewidth_935, mesh.thetaBE, mesh.detuning)

    eta = yb171.eta(excited_pop_935)
    excited_pop = yb171.excited_population_with_leakage(excited_pop_370, eta)

    other_data = {370: {"s0": mesh.s0,
                        "rabi": rabi_370,
                        "zeeman": mesh.zeeman,
                        "linewidth": eff_linewidth_370,
                        "pop": excited_pop_370},
                  935: {"s0": s0_935,
                        "rabi": rabi_935,
                        "linewidth": eff_linewidth_935,
                        "pop": excited_pop_935},
                  "eta": eta}

    return excited_pop, mesh, yb171, cart, other_data


def calculate_174_pop(detuning: Union[np.array, float],
                      I370: Union[np.array, float],
                      I935: Union[np.array, float],
                      thetaBE: Union[None, np.array, float]=None,
                      b_field: Union[None, PolarVector]=None,
                      e_field: Union[None, PolarVector]=None,
                      b_mag: Union[None, np.array, float]=None,
                      I370sat=None):
    cart = None

    if thetaBE is not None and b_mag is not None:
        pass
    elif e_field is not None and b_field is not None:
        thetaBE, cart = b_field.calculate_angle_between(e_field)
    else:
        raise ValueError("ThetaBE cannot be calculated. e_field, b_field or thetaBE, b_mag aren't supplied. "
                         "Either thetaBE and b_mag needs to be supplied or e_field and b_field.")

    variables = (thetaBE, detuning, I370, I935, b_field.r if b_field is not None else b_mag)

    mesh = GenMeshgrid(variables, ("thetaBE", "detuning", "I370", "I935", "bfieldmag"))
    mesh.gen_meshgrid()

    yb174 = Yb174()
    s0_370 = yb174.s0(mesh.I370, yb174.I370sat if I370sat is None else I370sat)
    rabi_370 = yb174.rabi_freq(s0_370, yb174.gamma_2S12_2P12)
    zeeman = yb174.zeeman_shift(mesh.bfieldmag)
    eff_linewidth_370 = yb174.effective_linewidth(mesh.thetaBE, rabi_370, zeeman, yb174.gamma_2S12_2P12)
    excited_pop_370 = yb174.excited_population_no_leakage(rabi_370, eff_linewidth_370, mesh.detuning)

    eta = yb174.eta()
    excited_pop = yb174.excited_population_with_leakage(excited_pop_370, eta)

    return excited_pop, mesh, yb174, cart


class GenerateTestData:

    def __init__(self, *args, **kwargs):
        self.excited_pop_174, self.mesh_174, self.yb174 = calculate_174_pop(*args, **kwargs)
        self.excited_pop_171, self.mesh_171, self.yb171 = calculate_171_pop(*args, **kwargs)

    def randomise(self, variance, seed=None):

        np.random.seed(seed)

        self.excited_pop_171 = self.excited_pop_174 + np.random.normal(loc=0, scale=np.sqrt(variance),
                                                                       size=len(self.excited_pop_171))
        self.excited_pop_174 = self.excited_pop_174 + np.random.normal(loc=0, scale=np.sqrt(variance),
                                                                       size=len(self.excited_pop_174))


class fitting_free_params:

    def __init__(self, br, btheta, bphi, er, etheta, ephi, I935):
        self.b_field = PolarVector(br, btheta, bphi)
        self.e_field = PolarVector(er, etheta, ephi)
        self.I935 = I935

    def fit(self, I370, detuning, b_mag, b_theta, b_phi, e_theta, Isat):
        """
        I370, detuning, b_mag and e_theta should be the independent variables. e_phi should be fixed
        and b_theta and Isat should be free parameters to be fitted.
        """
        self.e_field.theta = e_theta
        thetaBE = self.b_field.calculate_angle_between(self.e_field)
        calculate_171_pop(detuning, I370, self.I935, )






