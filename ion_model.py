import numpy as np
from numpy import pi as pi
from scipy.constants import h, c, hbar
from scipy.constants import physical_constants
from typing import Union, Tuple, List
from scipy.optimize import curve_fit
from scipy.special import factorial
import matplotlib.pyplot as plt
from dataclasses import dataclass
import inspect


@dataclass
class YbConstants:
	gamma_2S12_2P12_Hz: float = 19.7e6    # 1/s
	lambda_2S12_2P12: float = 369.5e-9          # m
	lambda_2D32_3D3212: float = 935.2e-9        # m
	gamma_2D32_3D3212_Hz: float = 4.2e6         # 1/ s
	zeeman_P_state_proportion: float = 1/3
	zeeman_S_state_proportion: float = 1
	branch_ratio_3D3212_2S12: float = 0.982
	hyperfine_splitting_2S12_Hz = 12.643e9  # 1/s
	hyperfine_splitting_2P12_Hz = 2.105e9   # 1/s
	hyperfine_splitting_2D32_Hz = 0.86e9    # radians/s
	hyperfine_splitting_3D32_Hz = 2.21e9    # radians/s

	optimal_thetaBE = np.arccos(1/np.sqrt(3))

	excited_lifetime_2P12 = 8.1e-9  #s
	excited_lifetime_3D32 = 37.7e-9  #s
	excited_lifetime_2D32 = 52.7e-3  #s

	zeeman_shift_Hz = 3.5e6


@dataclass
class Yb(YbConstants):
	basic_model: bool = False

	def __post_init__(self):
		self.gamma_2S12_2P12 = self.gamma_2S12_2P12_Hz * 2 * np.pi    # radians/s
		self.gamma_2D32_3D3212 = 2 * pi * self.gamma_2D32_3D3212_Hz   # radians / s
		self.mu_b = physical_constants['Bohr magneton'][0] / hbar  # kg m^2 s^-2
		self.branch_ratio_2P12_2D32 = 1 - self.branch_ratio_3D3212_2S12
		self.hyperfine_splitting_2S12 = 2 * pi * self.hyperfine_splitting_2S12_Hz # radians/s
		self.hyperfine_splitting_2P12 = 2 * pi * self.hyperfine_splitting_2P12_Hz   # radians/s
		self.hyperfine_splitting_2D32 = 2 * pi * self.hyperfine_splitting_2D32_Hz    # radians/s
		self.hyperfine_splitting_3D32 = 2 * pi * self.hyperfine_splitting_3D32_Hz    # radians/s

		self.optimal_thetaBE_171 = np.arccos(1/np.sqrt(3))
		self.optimal_thetaBE_174 = np.pi

		self.I370sat = pi * h * c * self.gamma_2S12_2P12 / ( 3 * self.lambda_2S12_2P12 ** 3)  # W/m^2
		self.I935sat = pi * h * c * self.gamma_2D32_3D3212 / (3 * self.lambda_2D32_3D3212 ** 3)  # W/m^2

		self.Gamma_2P12 = 1/self.excited_lifetime_2P12
		self.Gamma_3D32 = 1/self.excited_lifetime_3D32
		self.Gamma_2D32 = 1/self.excited_lifetime_2D32

	def zeeman_shift(self, B, g):
		return g * self.mu_b * B

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

	def update_internal_params(self, attributes_to_update: dict):
		for name, value in attributes_to_update.items():
			self.__setattr__(name, value)


@dataclass
class Yb171(Yb):

	EO_370_voltage: float = 0.5
	EO_angle_offset: float = np.arccos(1/np.sqrt(3))
	power_370_W: float = 5e-6
	power_370_sat_W: float = 1.8e-6
	power_935_W: float = 500e-6
	power_935_sat_W: float = 50e-6
	E370theta_rad: float = np.pi/2
	E370phi_rad: float = np.arccos(1/np.sqrt(3))
	E935theta_rad: float = np.pi/2
	E935phi_rad: float = np.arccos(1/np.sqrt(3))
	Btheta_rad: float = 0.0
	Bphi_rad: float = 0.0
	zeeman_shift_MHz: float = 3.5
	detuning_370_MHz: float = 5.0
	detuning_935_MHz: float = 0.0 
	photon_collection_efficiency: float = 0.015


	def __post_init__(self):
		super().__post_init__()
		self.E370phi_rad = self.EO_370_voltage * np.pi + self.EO_angle_offset
		self.s_370 = self.s0(self.power_370_W, self.power_370_sat_W) # converting this into a saturation parameter
		self.s_935 = self.s0(self.power_935_W, self.power_935_sat_W) # Converting this into a saturation parameter
		self.E370 = PolarVector(1, self.E370theta_rad, self.E370phi_rad) # Using vectors to describe the E and B fields
		self.E935 = PolarVector(1, self.E935theta_rad, self.E935phi_rad) #
		self.B = PolarVector(1, self.Btheta_rad, self.Bphi_rad)          #
		self.zeeman_shift_rad_s = 2 * np.pi * self.zeeman_shift_MHz * 1e6  # Convering to rad/s
		self.detuning_370 = 2 * np.pi * self.detuning_370_MHz * 1e6  # converting to rad/s
		self.detuning_935 = 2 * np.pi * self.detuning_935_MHz * 1e6  # converting to rad/s
		self.thetaBE370 = self.B.calculate_angle_between(self.E370)
		self.thetaB935 = self.B.calculate_angle_between(self.E935)
		self.optimal_thetaBE = self.optimal_thetaBE_171


	def effective_linewidth(self, thetaBE, rabi_freq, zeeman_shift, gamma):
		half_linewidth_squared = (gamma/2)**2
		cos2_thetaBE = np.cos(thetaBE)**2
		return half_linewidth_squared + \
			   rabi_freq**2/3 * cos2_thetaBE * (1 - 3 * cos2_thetaBE)/(1 + 3 * cos2_thetaBE) + \
			   cos2_thetaBE / (1 + 3 * cos2_thetaBE) * (rabi_freq ** 4 / (36 * zeeman_shift ** 2) + 4 * zeeman_shift ** 2)

	def excited_population_no_leakage(self, rabi_freq=None, effective_linewidth=None, thetaBE=None, detuning=None):
		if rabi_freq is None:
			rabi_freq = self.rabi_freq(self.s_370, self.gamma_2S12_2P12)
		if effective_linewidth is None:
			effective_linewidth = self.effective_linewidth(self.thetaB370, rabi_freq, self.zeeman_shift, self.gamma_2S12_2P12)
		if thetaBE is None:
			thetaBE = self.thetaBE370
		if detuning is None:
			detuning = self.detuning_370
		cos2_thetaBE = np.cos(thetaBE) ** 2
		sin2_thetaBE = np.sin(thetaBE) ** 2
		return 3/4 * cos2_thetaBE * sin2_thetaBE / (1 + 3 * cos2_thetaBE) * (rabi_freq ** 2 / 3) / \
			   (detuning**2 + effective_linewidth)

	def counts(self, 
			   photon_collection_efficiency=None, 
			   power_370_W = None, 
			   power_370_sat_W = None, 
			   E370theta_rad = None, 
			   EO_370_voltage = None,
			   EO_angle_offset = None,
			   Btheta_rad = None, 
			   Bphi_rad = None, 
			   E935theta_rad = None, 
			   E935phi_rad = None,
			   power_935_W = None,
			   power_935_sat_W = None,
			   zeeman_shift_MHz = None,
			   detuning_370_MHz = None,
			   detuning_935_MHz = None):
		"""
		Calculate the prediced counts from the ion given all of the possible 
		experimental configurations. This method will also acts as the fitting
		function for all other experiments.

		@params
		photon_collection_efficiency: float, the efficiency of photon collection (0<=eta<=1)
		power_370_W: float, the power being sent to the ion in Watts
		power_370_sat_W: float, the saturation power of the ion in Watts
		E370theta_rad: float, The theta Euler angle of the 370 laser electric field in radians
		EO_370_voltage: float, The Polarisation EO for the 370 laser controls the phi angle of the electric
							   field (the polarisation). This voltage is between 0 and 1, and rotates the 
							   polarisation from 0 to pi radians. This voltage is converted to E370phi
		EO_angle_offset: float, The input angle into the EO offsets the phi angle initially, so this is added to 
								E370 phi, and left as a free variable.
		E935theta_rad: float, the theta Euler angle of the 935 laser electric field in radians
		E935phi_rad: float, the phi Euler angle of the 935 laser electric field in radians
		Btheta_rad: float, the theta Euler angle of the B field vector at the ions position in radians
		Bphi_rad: float, the phi Euler angle of the B field vector at the ions position in radians
		zeeman_shift_MHz: float, the magnetic field induced Zeeman shift in units of MHz
		detuning_370_MHz: float, the detuning of the 370 laser from the excited state in units of MHz
		detuning_935_MHz: float, the detuning of the 935 laser from the excited state in units of MHz
		"""
		# updating all the internal state values
		if photon_collection_efficiency is not None:
			self.photon_collection_efficiency = photon_collection_efficiency
		if power_370_W is not None:
			self.power_370_W = power_370_W
		if power_370_sat_W is not None:
			self.power_370_sat_W = power_370_sat_W 
		if E370theta_rad is not None:
			self.E370theta_rad = E370theta_rad
		if EO_370_voltage is not None:
			self.EO_370_voltage = EO_370_voltage
		if EO_angle_offset is not None:
			self.EO_angle_offset = EO_angle_offset
		if Btheta_rad is not None:
			self.Btheta_rad = Btheta_rad
		if Bphi_rad is not None:
			self.Bphi_rad = Bphi_rad
		if E935theta_rad is not None:
			self.E935theta_rad = E935theta_rad 
		if E935phi_rad is not None:
			self.E935phi_rad = E935phi_rad
		if power_935_W is not None:
			self.power_935_W = power_935_W
		if power_935_sat_W is not None:
			self.power_935_sat_W = power_935_sat_W
		if zeeman_shift_MHz is not None:
			self.zeeman_shift_MHz = zeeman_shift_MHz
		if detuning_370_MHz is not None:
			self.detuning_370_MHz = detuning_370_MHz
		if detuning_935_MHz is not None:
			self.detuning_935_MHz = detuning_935_MHz

		self.E370phi_rad = self.EO_370_voltage * np.pi*0.90 + self.EO_angle_offset
		self.s_370 = self.s0(self.power_370_W, self.power_370_sat_W) # converting this into a saturation parameter
		self.s_935 = self.s0(self.power_935_W, self.power_935_sat_W) # Converting this into a saturation parameter
		self.E370 = PolarVector(1,self.E370theta_rad, self.E370phi_rad) # Using vectors to describe the E and B fields
		self.E935 = PolarVector(1, self.E935theta_rad, self.E935phi_rad) #
		self.B = PolarVector(1, self.Btheta_rad, self.Bphi_rad)          #
		self.thetaB370 = self.B.calculate_angle_between(self.E370)
		self.thetaB935 = self.B.calculate_angle_between(self.E935)
		self.zeeman_shift_rad_s = 2 * np.pi * self.zeeman_shift_MHz * 1e6  # Convering to rad/s
		self.detuning_370 = 2 * np.pi * self.detuning_370_MHz * 1e6  # converting to rad/s
		self.detuning_935 = 2 * np.pi * self.detuning_935_MHz * 1e6  # converting to rad/s
		excited_pop, mesh, yb171, other_data = calculate_171_pop(b_field=self.B, 
																 e_field_370=self.E370,
																 e_field_935=self.E935,
																 s_370=self.s_370, 
																 s_935=self.s_935, 
																 zeeman=self.zeeman_shift_rad_s,
																 detuning370=self.detuning_370,
																 detuning935=self.detuning_935, make_mesh=False)
		if self.basic_model:
			excited_pop = other_data[370]['pop']
		total_counts_per_s = excited_pop * self.gamma_2S12_2P12 * self.photon_collection_efficiency
		return total_counts_per_s

	def leakage_rate_of_bright_state(self):
		"""Leakage rate of the 2S_1/2 <-> 2P_1/2 transition into the D state
		This assumes you have already updated all the attributes of the ion
		model, it just returns the leakage rate based on the internal attributes.
		"""
		rabi_rate = self.rabi_freq(self.s_370, self.gamma_2S12_2P12)
		effective_lw = self.effective_linewidth(self.thetaBE370, rabi_rate, self.zeeman_shift_rad_s, self.gamma_2S12_2P12)
		return 2 / 27 * self.gamma_2S12_2P12  * (rabi_rate/(2*self.hyperfine_splitting_2P12))**2 * (1-self.excited_population_no_leakage(rabi_rate, effective_lw, self.thetaB370, self.detuning_370))

	def leakage_rate_of_dark_state(self):
		"""Leakage rate of the dark state back into the cooling cycle"""
		rabi_rate = self.rabi_freq(self.s_370, self.gamma_2S12_2P12)
		return 2/9 * self.gamma_2S12_2P12 * (rabi_rate/(2*(self.hyperfine_splitting_2S12 + self.hyperfine_splitting_2P12)))**2


@dataclass
class Yb174(Yb):
	EO_370_voltage: float = 0.7
	EO_angle_offset: float = -np.arccos(1/np.sqrt(3))
	power_370_W: float = 5e-6
	power_370_sat_W: float = 1.8e-6
	power_935_W: float = 500e-6
	power_935_sat_W: float = 50e-6
	E370theta_rad: float = np.pi/2
	E370phi_rad: float = np.pi
	E935theta_rad: float = np.pi/2
	E935phi_rad: float = np.pi
	Btheta_rad: float = 0.0
	Bphi_rad: float = 0.0
	zeeman_shift_MHz: float = 3.5
	detuning_370_MHz: float = 5.0
	detuning_935_MHz: float = 0.0 
	photon_collection_efficiency: float = 0.015


	def __post_init__(self):
		super().__post_init__()
		self.E370phi_rad = self.EO_370_voltage * np.pi + self.EO_angle_offset
		self.s_370 = self.s0(self.power_370_W, self.power_370_sat_W) # converting this into a saturation parameter
		self.s_935 = self.s0(self.power_935_W, self.power_935_sat_W) # Converting this into a saturation parameter
		self.E370 = PolarVector(1, self.E370theta_rad, self.E370phi_rad) # Using vectors to describe the E and B fields
		self.E935 = PolarVector(1, self.E935theta_rad, self.E935phi_rad) #
		self.B = PolarVector(1, self.Btheta_rad, self.Bphi_rad)          #
		self.zeeman_shift_rad_s = 2 * np.pi * self.zeeman_shift_MHz * 1e6  # Convering to rad/s
		self.detuning_370 = 2 * np.pi * self.detuning_370_MHz * 1e6  # converting to rad/s
		self.detuning_935 = 2 * np.pi * self.detuning_935_MHz * 1e6  # converting to rad/s
		self.thetaBE370 = self.B.calculate_angle_between(self.E370)
		self.thetaB935 = self.B.calculate_angle_between(self.E935)
		self.optimal_thetaBE = self.optimal_thetaBE_171

	def effective_linewidth(self, thetaBE, rabi_freq, zeeman_shift, gamma):
		cos2 = np.cos(thetaBE)**2
		gamma92 = 9 * gamma ** 2
		zeeman162 = 16 * zeeman_shift ** 2 / gamma92
		return rabi_freq**2/6 + gamma**2/4 * (1 + zeeman162) *\
			   (1 + 64 * zeeman_shift**2 / gamma92) / (1 + zeeman162 * (3 * cos2 + 1))

	def excited_population_no_leakage(self, rabi_freq, thetaBE, effective_linewidth, detuning):
		return 1/2 * (rabi_freq**2 / 6) / (detuning ** 2 + effective_linewidth)

	def counts(self, 
			   photon_collection_efficiency=None, 
			   power_370_W = None, 
			   power_370_sat_W = None, 
			   E370theta_rad = None, 
			   EO_370_voltage = None,
			   EO_angle_offset = None,
			   Btheta_rad = None, 
			   Bphi_rad = None, 
			   E935theta_rad = None, 
			   E935phi_rad = None,
			   power_935_W = None,
			   power_935_sat_W = None,
			   zeeman_shift_MHz = None,
			   detuning_370_MHz = None,
			   detuning_935_MHz = None):
		"""
		Calculate the prediced counts from the ion given all of the possible 
		experimental configurations. This method will also acts as the fitting
		function for all other experiments.

		@params
		photon_collection_efficiency: float, the efficiency of photon collection (0<=eta<=1)
		power_370_W: float, the power being sent to the ion in Watts
		power_370_sat_W: float, the saturation power of the ion in Watts
		E370theta_rad: float, The theta Euler angle of the 370 laser electric field in radians
		EO_370_voltage: float, The Polarisation EO for the 370 laser controls the phi angle of the electric
							   field (the polarisation). This voltage is between 0 and 1, and rotates the 
							   polarisation from 0 to pi radians. This voltage is converted to E370phi
		EO_angle_offset: float, The input angle into the EO offsets the phi angle initially, so this is added to 
								E370 phi, and left as a free variable.
		E935theta_rad: float, the theta Euler angle of the 935 laser electric field in radians
		E935phi_rad: float, the phi Euler angle of the 935 laser electric field in radians
		Btheta_rad: float, the theta Euler angle of the B field vector at the ions position in radians
		Bphi_rad: float, the phi Euler angle of the B field vector at the ions position in radians
		zeeman_shift_MHz: float, the magnetic field induced Zeeman shift in units of MHz
		detuning_370_MHz: float, the detuning of the 370 laser from the excited state in units of MHz
		detuning_935_MHz: float, the detuning of the 935 laser from the excited state in units of MHz
		"""
		if photon_collection_efficiency is not None:
			self.photon_collection_efficiency = photon_collection_efficiency
		if power_370_W is not None:
			self.power_370_W = power_370_W
		if power_370_sat_W is not None:
			self.power_370_sat_W = power_370_sat_W 
		if E370theta_rad is not None:
			self.E370theta_rad = E370theta_rad
		if EO_370_voltage is not None:
			self.EO_370_voltage = EO_370_voltage
		if EO_angle_offset is not None:
			self.EO_angle_offset = EO_angle_offset % np.pi
		if Btheta_rad is not None:
			self.Btheta_rad = Btheta_rad
		if Bphi_rad is not None:
			self.Bphi_rad = Bphi_rad
		if E935theta_rad is not None:
			self.E935theta_rad = E935theta_rad 
		if E935phi_rad is not None:
			self.E935phi_rad = E935phi_rad
		if power_935_W is not None:
			self.power_935_W = power_935_W
		if power_935_sat_W is not None:
			self.power_935_sat_W = power_935_sat_W
		if zeeman_shift_MHz is not None:
			self.zeeman_shift_MHz = zeeman_shift_MHz
		if detuning_370_MHz is not None:
			self.detuning_370_MHz = detuning_370_MHz
		if detuning_935_MHz is not None:
			self.detuning_935_MHz = detuning_935_MHz
		
		self.E370phi_rad = self.EO_370_voltage * np.pi + self.EO_angle_offset
		self.s_370 = self.s0(self.power_370_W, self.power_370_sat_W) # converting this into a saturation parameter
		self.s_935 = self.s0(self.power_935_W, self.power_935_sat_W) # Converting this into a saturation parameter
		self.E370 = PolarVector(1,self.E370theta_rad, self.E370phi_rad) # Using vectors to describe the E and B fields
		self.E935 = PolarVector(1, self.E935theta_rad, self.E935phi_rad) #
		self.B = PolarVector(1, self.Btheta_rad, self.Bphi_rad)          #
		self.thetaB370 = self.B.calculate_angle_between(self.E370)
		self.thetaB935 = self.B.calculate_angle_between(self.E935)
		self.zeeman_shift_rad_s = 2 * np.pi * self.zeeman_shift_MHz * 1e6  # Convering to rad/s
		self.detuning_370 = 2 * np.pi * self.detuning_370_MHz * 1e6  # converting to rad/s
		self.detuning_935 = 2 * np.pi * self.detuning_935_MHz * 1e6  # converting to rad/s
		excited_pop, mesh, yb174, other_data = calculate_174_pop(b_field=self.B, 
																 e_field_370=self.E370,
																 e_field_935=self.E935,
																 s_370=self.s_370, 
																 s_935=self.s_935, 
																 zeeman=self.zeeman_shift_rad_s,
																 detuning370=self.detuning_370,
																 detuning935=self.detuning_935, make_mesh=False)
		if self.basic_model:
			excited_pop = other_data['excited_pop_370']
		total_counts_per_s = excited_pop * self.gamma_2S12_2P12 * self.photon_collection_efficiency
		return total_counts_per_s
	
	def leakage_rate_of_bright_state(self):
		"""Leakage rate of the 2S_1/2 <-> 2P_1/2 transition into the D state
		This assumes you have already updated all the attributes of the ion
		model, it just returns the leakage rate based on the internal attributes.
		"""
		rabi_rate = self.rabi_freq(self.s_370, self.gamma_2S12_2P12)
		effective_lw = self.effective_linewidth(self.thetaBE370, rabi_rate, self.zeeman_shift_rad_s, self.gamma_2S12_2P12)
		return 2 / 27 * self.gamma_2S12_2P12  * (rabi_rate/(2*self.hyperfine_splitting_2P12))**2 * (1-self.excited_population_no_leakage(rabi_rate, effective_lw, self.thetaB370, self.detuning_370))

	def leakage_rate_of_dark_state(self):
		"""Leakage rate of the dark state back into the cooling cycle"""
		rabi_rate = self.rabi_freq(self.s_370, self.gamma_2S12_2P12)
		return 2/9 * self.gamma_2S12_2P12 * (rabi_rate/(2*(self.hyperfine_splitting_2S12 + self.hyperfine_splitting_2P12)))**2


class PolarVectorGeneral:
	def __init__(self,
				 r: Union[float, None] = None,
				 theta: Union[float, None] = None,
				 phi: Union[float, None] = None,
				 cartesian: Union[tuple, None] = None):
		if cartesian is not None:
			x, y, z = cartesian
			self.v = np.array([[0, x], [0, y], [0, z]], dtype=np.float64)
			self.r = None
			self.theta = None
			self.phi = None
			self.convert_to_spherical()
		elif any([r is None, theta is None, phi is None]):
			raise ValueError
		else:
			self.r = r
			self.theta = theta
			self.phi = phi
			self.pv = np.array([[self.r], [self.theta], [self.phi]], dtype=object)
			self.convert_to_cartesian()

	@staticmethod
	def Rx(theta):
		Rx = np.array([[1,             0,              0], 
					   [0, np.cos(theta), -np.sin(theta)], 
					   [0, np.sin(theta),  np.cos(theta)]])
		return Rx
	
	@staticmethod
	def Ry(theta):
		Ry = np.array([[np.cos(theta), 0,  np.sin(theta)], 
					   [0,             1,              0], 
					   [-np.sin(theta), 0, np.cos(theta)]])
		return Ry
	
	@staticmethod
	def Rz(theta):
		Rz = np.array([[np.cos(theta), -np.sin(theta), 0], 
					   [np.sin(theta),  np.cos(theta), 0], 
					   [0,          0,                 1]])
		return Rz
	
	def update_pv(self):
		self.pv = np.array([[self.r], [self.theta], [self.phi]])

	def update(self, r, theta, phi):
		self.r = r
		self.theta = theta
		self.phi = phi

	def rotate_theta_by(self, radians):
		self.theta += radians
		self.convert_to_cartesian()
		self.update_pv()

	def rotate_phi_by(self, radians):
		self.phi += radians
		self.convert_to_cartesian()
		self.update_pv()

	def change_mag(self, length):
		self.r += length
		self.convert_to_cartesian()
		self.update_pv()

	def scale_mag(self, scale):
		self.r *= scale
		self.convert_to_cartesian()
		self.update_pv()

	def convert_to_cartesian(self):
		x = self.r * np.sin(self.phi) * np.cos(self.theta)
		y = self.r * np.sin(self.phi) * np.sin(self.theta)
		z = self.r * np.cos(self.phi)
		self.v = np.array([[0, x], [0, y], [0, z]], dtype=object)
	
	def convert_to_spherical(self):
		vec = self.v[:, 1] - self.v[:, 0]
		self.r = np.sqrt(sum(vec**2))
		self.theta = np.arctan2(vec[1], vec[0])
		self.phi = np.arctan2(np.sqrt(vec[0]**2 + vec[1]**2), vec[2])
		self.pv = np.array([[self.r], [self.theta], [self.phi]])
	
	def rotate_about_x(self, theta):
		self.v = self.Rx(theta) @ self.v
		self.convert_to_spherical()
	
	def rotate_about_y(self, theta):
		self.v = self.Ry(theta) @ self.v
		self.convert_to_spherical()

	def rotate_about_z(self, theta):
		self.v = self.Rz(theta) @ self.v
		self.convert_to_spherical()
	
	def translate(self, translation):
		self.v = self.v + translation

	def flip(self):
		self.v = np.flip(self.v, 1)
		self.convert_to_spherical()
	

def unwrap_angle(angles):
	""" Assumed that the angle should be going in one direction at all times """
	direction = np.sign(angles[1:] - angles[:-1])
	direction = np.insert(direction, 0, direction[0])
	angles = angles * -1 * direction
	
	# direction = np.abs((direction[1:] - direction[:-1]))
	# change_index = np.argwhere(direction == 2)[:, 0]
	# even = len(change_index) % 2 == 0

	# if len(change_index) == 0:
		# return angles
	
	# skip = 0
	# for i, ind in enumerate(change_index):
		# if i < len(change_index) - 1 and skip:
			# angles[ind: change_index[i+1] + 1] = angles[ind] + (pi - angles[ind: change_index[i+1] + 1])
			# skip = 0
		# elif not even and not skip:
			# angles[ind:] = angles[ind] + (pi - angles[ind:])
		# elif not skip:
			# skip = 1

	return angles


class PolarVector(PolarVectorGeneral):

	def calculate_angle_between(self, other_vector: PolarVectorGeneral):
		vec1 = self.v[:, 1] - self.v[:, 0]
		vec2 = other_vector.v[:, 1] - other_vector.v[:, 0]
		dot_product = np.dot(vec1, vec2)
		angle = np.arccos(dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
		return angle


	def arb_rot_mat(self, vector: PolarVectorGeneral, angle):
		vec1 = np.array(self.v[:, 1] - self.v[:, 0], dtype=np.float64)
		vec2 = np.array(vector.v[:, 1] - vector.v[:, 0], dtype=np.float64)
		cross_product = np.cross(vec1, vec2)
		rotation_matrix = np.array([[np.cos(angle) + cross_product[0]**2*(1-np.cos(angle)), cross_product[0]*cross_product[1]*(1-np.cos(angle)) - cross_product[2]*np.sin(angle), cross_product[0]*cross_product[2]*(1-np.cos(angle)) + cross_product[1]*np.sin(angle)],
								[cross_product[1]*cross_product[0]*(1-np.cos(angle)) + cross_product[2]*np.sin(angle), np.cos(angle) + cross_product[1]**2*(1-np.cos(angle)), cross_product[1]*cross_product[2]*(1-np.cos(angle)) - cross_product[0]*np.sin(angle)],
								[cross_product[2]*cross_product[0]*(1-np.cos(angle)) - cross_product[1]*np.sin(angle), cross_product[2]*cross_product[1]*(1-np.cos(angle)) + cross_product[0]*np.sin(angle), np.cos(angle) + cross_product[2]**2*(1-np.cos(angle))]])
		return rotation_matrix
	
	def rotate_with_rotation_matrix(self, rotation_matix):
		self.v = np.dot(rotation_matix, self.v)
		self.convert_to_spherical()


class CartVector(PolarVector):

	def __init__(self, x: float, y: float, z: float):
		self.v = np.array([[0, x], [0, y], [0, z]])
		vec = self.v[:, 1] - self.v[:, 0]
		self.r = np.sqrt(sum(vec**2))
		self.theta = np.arccos(vec[2]/ self.r)
		self.phi = np.arctan2(vec[1], vec[0])
		self.pv = np.array([[self.r], [self.theta], [self.phi]])


class GenMeshgrid:

	def __init__(self, poss_arrays: tuple, names: tuple):
		self.names = names
		for i, name in enumerate(self.names):
			self.__setattr__(name, poss_arrays[i])
		self.variables_forming_meshgrid = (var for var in poss_arrays if type(var) == np.ndarray)
		self.mesh_order = {name: len(poss_arrays[i]) for i, name in enumerate(names) if type(poss_arrays[i]) == np.ndarray}
		self.variables_names = (names[i] for i, var in enumerate(poss_arrays) if type(var) == np.ndarray)
		self.mesh_shape = tuple(self.mesh_order.values())

	def gen_meshgrid(self):
		meshgrids = np.meshgrid(*self.variables_forming_meshgrid)
		for i, name in enumerate(self.variables_names):
			self.__setattr__(name, meshgrids[i])


def calculate_171_pop(detuning370: Union[np.array, float] = 0,
					  detuning935: Union[np.array, float] = 0,
					  I935: Union[np.array, float, None] = None,
					  I370: Union[np.array, float, None] = None,
					  thetaBE370: Union[np.array, float, None] = None,
					  thetaBE935: Union[np.array, float, None] = None,
					  b_field: Union[PolarVector, None] = None,
					  e_field_370: Union[PolarVector, None] = None,
					  e_field_935: Union[PolarVector, None] = None,
					  b_mag: Union[np.array, float, None] = None,
					  zeeman: Union[np.array, float, None] = None,
					  s_370: Union[np.array, float, None] = None,
					  s_935: Union[np.array, float, None] = None,
					  make_mesh=True) -> Tuple[np.ndarray, np.ndarray, Yb171, dict]:

	yb171 = Yb171()

	if thetaBE370 is not None and b_mag is not None:
		pass
	elif e_field_370 is not None and b_field is not None:
		thetaBE370 = b_field.calculate_angle_between(e_field_370)
		b_mag = b_field.r
	elif thetaBE370 is not None and zeeman is not None:
		pass
	else:
		raise ValueError("ThetaBE370 cannot be calculated. e_field_370, b_field or thetaBE370, b_mag aren't supplied. "
						 "Either thetaBE370 and b_mag needs to be supplied or e_field_370 and b_field.")
	

	if thetaBE935 is not None and b_mag is not None:
		pass
	elif e_field_935 is not None and b_field is not None:
		thetaBE935 = b_field.calculate_angle_between(e_field_935)
		b_mag = b_field.r
	elif thetaBE935 is not None and zeeman is not None:
		pass
	else:
		raise ValueError("ThetaBE370 cannot be calculated. e_field_370, b_field or thetaBE370, b_mag aren't supplied. "
						 "Either thetaBE370 and b_mag needs to be supplied or e_field_370 and b_field.")


	if I370 is not None:
		s0_370 = yb171.s0(I370, yb171.I370sat)
	elif s_370 is not None:
		s0_370 = s_370
	else:
		raise ValueError("I370 or s_370 need to be passed")
	
	if I935 is not None:
		s0_935 = yb171.s0(I935, yb171.I935sat)
	elif s_935 is not None:
		s0_935 = s_935
	else:
		raise ValueError("I935 or s_935 need to be passed")

	zeeman = yb171.zeeman_shift(b_mag, 1) if zeeman is None else zeeman

	variables = (thetaBE370, thetaBE935, detuning370, detuning935, s0_370, s0_935, zeeman)

	mesh = GenMeshgrid(variables, ("thetaBE370", "thetaBE935", "detuning370", "detuning935", "s0_370", "s0_935", "zeeman"))
	if make_mesh:
		mesh.gen_meshgrid()

	rabi_370 = yb171.rabi_freq(mesh.s0_370, yb171.gamma_2S12_2P12)
	eff_linewidth_370 = yb171.effective_linewidth(mesh.thetaBE370, rabi_370, mesh.zeeman, yb171.gamma_2S12_2P12)
	excited_pop_370 = yb171.excited_population_no_leakage(rabi_370, eff_linewidth_370, mesh.thetaBE370, mesh.detuning370)
  
	rabi_935 = yb171.rabi_freq(mesh.s0_935, yb171.gamma_2D32_3D3212)
	eff_linewidth_935 = yb171.effective_linewidth(mesh.thetaBE935, rabi_935, mesh.zeeman, yb171.gamma_2D32_3D3212)
	excited_pop_935 = yb171.excited_population_no_leakage(rabi_935, eff_linewidth_935, mesh.thetaBE935, mesh.detuning935)

	eta = yb171.eta(excited_pop_935)
	excited_pop = yb171.excited_population_with_leakage(excited_pop_370, eta)

	other_data = {370: {"s0": mesh.s0_370,
						"rabi": rabi_370,
						"zeeman": mesh.zeeman,
						"linewidth": eff_linewidth_370,
						"pop": excited_pop_370},
				  935: {"s0": mesh.s0_935,
						"rabi": rabi_935,
						"linewidth": eff_linewidth_935,
						"pop": excited_pop_935},
				  "eta": eta}

	return excited_pop, mesh, yb171, other_data


def calculate_174_pop(detuning370: Union[np.array, float] = 0,
					  detuning935: Union[np.array, float] = 0,
					  I935: Union[np.array, float, None] = None,
					  I370: Union[np.array, float, None] = None,
					  thetaBE370: Union[np.array, float, None] = None,
					  thetaBE935: Union[np.array, float, None] = None,
					  b_field: Union[PolarVector, None] = None,
					  e_field_370: Union[PolarVector, None] = None,
					  e_field_935: Union[PolarVector, None] = None,
					  b_mag: Union[np.array, float, None] = None,
					  zeeman: Union[np.array, float, None] = None,
					  s_370: Union[np.array, float, None] = None,
					  s_935: Union[np.array, float, None] = None,
					  make_mesh=True):

	yb174 = Yb174()

	if thetaBE370 is not None and b_mag is not None:
		pass
	elif e_field_370 is not None and b_field is not None:
		thetaBE370 = b_field.calculate_angle_between(e_field_370)
		b_mag = b_field.r
	elif thetaBE370 is not None and zeeman is not None:
		pass
	else:
		raise ValueError("ThetaBE370 cannot be calculated. e_field_370, b_field or thetaBE370, b_mag aren't supplied. "
						 "Either thetaBE370 and b_mag needs to be supplied or e_field_370 and b_field.")
	

	if thetaBE935 is not None and b_mag is not None:
		pass
	elif e_field_935 is not None and b_field is not None:
		thetaBE935 = b_field.calculate_angle_between(e_field_935)
		b_mag = b_field.r
	elif thetaBE935 is not None and zeeman is not None:
		pass
	else:
		raise ValueError("ThetaBE370 cannot be calculated. e_field_370, b_field or thetaBE370, b_mag aren't supplied. "
						 "Either thetaBE370 and b_mag needs to be supplied or e_field_370 and b_field.")


	if I370 is not None:
		s0_370 = yb174.s0(I370, yb174.I370sat)
	elif s_370 is not None:
		s0_370 = s_370
	else:
		raise ValueError("I370 or s_370 need to be passed")
	
	if I935 is not None:
		s0_935 = yb174.s0(I935, yb174.I935sat)
	elif s_935 is not None:
		s0_935 = s_935
	else:
		raise ValueError("I935 or s_935 need to be passed")

	zeeman = yb174.zeeman_shift(b_mag, 1) if zeeman is None else zeeman

	variables = (thetaBE370, thetaBE935, detuning370, detuning935, s0_370, s0_935, zeeman)

	mesh = GenMeshgrid(variables, ("thetaBE370", "thetaBE935", "detuning370", "detuning935", "s0_370", "s0_935", "zeeman"))
	if make_mesh:
		mesh.gen_meshgrid()

	rabi_370 = yb174.rabi_freq(mesh.s0_370, yb174.gamma_2S12_2P12)
	eff_linewidth_370 = yb174.effective_linewidth(mesh.thetaBE370, rabi_370, mesh.zeeman, yb174.gamma_2S12_2P12)
	excited_pop_370 = yb174.excited_population_no_leakage(rabi_370, mesh.thetaBE370, eff_linewidth_370, mesh.detuning370)

	rabi_935 = yb174.rabi_freq(mesh.s0_935, yb174.gamma_2D32_3D3212)
	eff_linewidth_935 = yb174.effective_linewidth(mesh.thetaBE935, rabi_935, mesh.zeeman, yb174.gamma_2D32_3D3212)
	excited_pop_935 = yb174.excited_population_no_leakage(rabi_935, mesh.thetaBE935, eff_linewidth_935, mesh.detuning935)

	eta = yb174.eta(excited_pop_935)
	excited_pop = yb174.excited_population_with_leakage(excited_pop_370, eta)

	other_data = {370: {"s0": mesh.s0_370,
						"rabi": rabi_370,
						"zeeman": mesh.zeeman,
						"linewidth": eff_linewidth_370,
						"pop": excited_pop_370},
				  935: {"s0": mesh.s0_935,
						"rabi": rabi_935,
						"linewidth": eff_linewidth_935,
						"pop": excited_pop_935},
				  "eta": eta}

	return excited_pop, mesh, yb174, other_data


class GenerateTestData:

	def __init__(self, *args, **kwargs):
		self.excited_pop_174, self.mesh_174, self.yb174, self.other_174 = calculate_174_pop(*args, **kwargs)
		self.excited_pop_171, self.mesh_171, self.yb171, self.other_171 = calculate_171_pop(*args, **kwargs)

	def randomise(self, variance, seed=None):

		np.random.seed(seed)

		self.excited_pop_171 = self.excited_pop_171 + np.random.normal(loc=0, scale=np.sqrt(variance),
																	   size=len(self.excited_pop_171))
		self.excited_pop_174 = self.excited_pop_174 + np.random.normal(loc=0, scale=np.sqrt(variance),
																	   size=len(self.excited_pop_174))


class FitFreeParams:

	"""This class should use the calculate_pop functions for 171 and 174 to fit free parameters.
	We will assume we have some known parameters (the ones were sweeping) and the rest are free or
	constant. Free parameters should be left blank, """

	def __init__(self, x, y, 
				 x_variable_names: list, 
				 yb_model,
				 all_variables=("thetaBE370", "s_370", "detuning370", "detuning935", "s_935", "zeeman", "thetaBE935")):
		# check if x_variable_names is in all_variables
		if not all([name in all_variables for name in x_variable_names]):
			raise ValueError(f"x_variable_names must match exactly the names {all_variables}. Received names were {x_variable_names}.")
		self.free_params = list(all_variables)
		for x_name in x_variable_names:
			self.free_params.remove(x_name)

		ordered_variable_names = list(x_variable_names)
		ordered_variable_names.extend(self.free_params)
		self.ordered_variable_names = tuple(ordered_variable_names)
		self.func = yb_model
		self.x = x
		self.y = y

	def fit_func(self, x: tuple, *args):
		"""
		I370, detuning, b_mag and e_theta should be the independent variables. e_phi should be fixed
		and b_theta and Isat should be free parameters to be fitted.
		"""
		listx = list(x)
		listx.extend(list(args))
		kwargs = {name: listx[i] for i, name in enumerate(self.ordered_variable_names)}
		kwargs['make_mesh'] = False
		y, _, _, _ = self.func(**kwargs)
		return y
	
	def fit(self, p0=None, sigma=None, **kwargs):
		popt, pcov = curve_fit(self.fit_func, self.x, self.y, p0=p0, sigma=sigma, **kwargs)
		return popt, pcov


class FitCounts:

	def __init__(self, independent_variables: dict, counts: np.ndarray, fit_function):
		"""
		This class handles fitting any function with many variables, where you can simply input what
		variables are being held constant (independent variables), while the rest are treated as free
		parameters to be fitted. 
		"""
		all_variables = inspect.getargspec(fit_function).args
		if 'self' in all_variables:
			all_variables.remove('self')

		names_of_independent_variables = list(independent_variables.keys())

		if not all([name in all_variables for name in names_of_independent_variables]):
			raise ValueError(f"x_variable_names must match exactly the names {all_variables}. Received names were {names_of_independent_variables}.")
		
		# Determine the free parameters to be fitted
		self.free_params = [var for var in all_variables if var not in names_of_independent_variables]
		
		self.counts = counts

		self.function = fit_function
		self.independent_variables = independent_variables
	
	def fit_func(self, *free_params):
		"""
		This function wraps the original function to use with curve_fit.
		It holds certain variables constant and lets the free parameters vary.
		
		:param free_params: The free parameters to be fitted.
		:return: The function evaluated with the free parameters and fixed independent variables.
		"""
		# Combine free parameters with fixed independent variables
		free_params = list(free_params)
		free_params.pop(0)
		all_params = {**self.independent_variables}
		all_params.update(dict(zip(self.free_params, free_params)))
		
		return self.function(**all_params)

	
	def fit(self, p0: dict=None, sigma: list=None, **curve_fit_kwargs):
		"""
		Fits the function to the data using curve_fit.
		
		:param p0: Initial guess for the free parameters.
		:param sigma: Optional uncertainties for the dependent variable.
		:param curve_fit_kwargs: Additional arguments to pass to curve_fit.
		:return: Optimal values for the free parameters and the covariance matrix.
		"""
		# Initial guess for the free parameters
		if p0 is None:
			p0 = [1.0] * len(self.free_params)
		else:
			p0 = [p0[name] for name in self.free_params]

		# Perform the curve fitting
		popt, pcov = curve_fit(self.fit_func, np.arange(len(self.counts)), self.counts, p0=p0, sigma=sigma, **curve_fit_kwargs)
		popt = {free_param_name: pop for free_param_name, pop in zip(self.free_params, popt)}
		pcov = {free_param_name: pop for free_param_name, pop in zip(self.free_params, pcov)}
		all_params = {**self.independent_variables}
		all_params.update(popt)
		self.all_params = all_params
		return popt, pcov





def calc_171_pop_using_vectors(Er=0, Ephi=0, Etheta=0, Br=0, Bphi=0, Btheta=0, **model_kwargs):
	E = PolarVector(Er, Etheta, Ephi)
	B = PolarVector(Br, Btheta, Bphi)
	excited_pop, mesh, yb171, other_data = calculate_171_pop(b_field=B, e_field_370=E, **model_kwargs)
	return excited_pop, mesh, yb171, other_data


def counts_171(efficiency, *args, **kwargs):
	excited_pop, mesh, yb171, other_data = calculate_171_pop(*args, **kwargs)
	return excited_pop * yb171.gamma_2S12_2P12 * efficiency, mesh, yb171, other_data

def counts_174(efficiency, *args, **kwargs):
	excited_pop, mesh, yb174, other_data = calculate_174_pop(*args, **kwargs)
	return excited_pop * yb174.gamma_2S12_2P12 * efficiency, mesh, yb174, other_data 

class FitCounts:

	def __init__(self, independent_variables: dict, counts: np.ndarray, fit_function):
		"""
		This class handles fitting any function with many variables, where you can simply input what
		variables are being held constant (independent variables), while the rest are treated as free
		parameters to be fitted. 
		"""
		all_variables = inspect.getargspec(fit_function).args
		if 'self' in all_variables:
			all_variables.remove('self')

		names_of_independent_variables = list(independent_variables.keys())

		if not all([name in all_variables for name in names_of_independent_variables]):
			check = [name in all_variables for name in names_of_independent_variables]
			print("Name not matching", [name for found, name in zip(check, names_of_independent_variables) if not found])
			raise ValueError(f"x_variable_names must match exactly the names {all_variables}. Received names were {names_of_independent_variables}.")
		
		# Determine the free parameters to be fitted
		self.free_params = [var for var in all_variables if var not in names_of_independent_variables]
		
		self.counts = counts

		self.function = fit_function
		self.independent_variables = independent_variables
	
	def fit_func(self, *free_params):
		"""
		This function wraps the original function to use with curve_fit.
		It holds certain variables constant and lets the free parameters vary.
		
		:param free_params: The free parameters to be fitted.
		:return: The function evaluated with the free parameters and fixed independent variables.
		"""
		# Combine free parameters with fixed independent variables
		free_params = list(free_params)
		free_params.pop(0)
		all_params = {**self.independent_variables}
		all_params.update(dict(zip(self.free_params, free_params)))
		
		return self.function(**all_params)

	
	def fit(self, p0: dict=None, sigma: list=None, **curve_fit_kwargs):
		"""
		Fits the function to the data using curve_fit.
		
		:param p0: Initial guess for the free parameters.
		:param sigma: Optional uncertainties for the dependent variable.
		:param curve_fit_kwargs: Additional arguments to pass to curve_fit.
		:return: Optimal values for the free parameters and the covariance matrix.
		"""
		# Initial guess for the free parameters
		if p0 is None:
			p0 = [1.0] * len(self.free_params)
		else:
			p0 = [p0[name] for name in self.free_params]

		# Perform the curve fitting
		popt, pcov = curve_fit(self.fit_func, np.arange(len(self.counts)), self.counts, p0=p0, sigma=sigma, **curve_fit_kwargs)
		popt = {free_param_name: pop for free_param_name, pop in zip(self.free_params, popt)}
		all_params = {**self.independent_variables}
		error = {free_param_name: er for free_param_name, er in zip(self.free_params, np.sqrt(np.diag(pcov)))}
		all_params.update(popt)
		self.all_params = all_params
		return popt, error, pcov



def Lorentzian(x, x0, Gamma):
	half_Gamma = Gamma/2
	return 1/np.pi * half_Gamma / ((x-x0)**2 + half_Gamma**2)


def poissonian(k, lam):
	return np.exp(-lam) * lam ** k / factorial(k)





