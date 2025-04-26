
from ..._globals import MAX_SF_RADIUS
from .utils import gaussian, modified_exponential
from .. import inputs
from .expifr import expifr
import math as m
import vice
# from ..gasflows import logistic_betaphiin


class GSE(gaussian):

	def __init__(self, radius, dr = 0.1, mgas = inputs.GSE_GAS_MASS,
		fe_h = inputs.GSE_FEH, o_h = inputs.GSE_OH, mg_h = inputs.GSE_MGH,
		t_acc = inputs.GSE_T_ACC, sigma_time = inputs.GSE_SIGMA_TIME):
		# r_acc = 10, sigma_radius = 1):

		self.radius = radius
		self.mgas = mgas
		self.fe_h = fe_h
		self.o_h = o_h
		self.mg_h = mg_h
		self.t_acc = t_acc
		self.sigma_time = sigma_time
		# self.r_acc = r_acc
		# self.sigma_radius = sigma_radius
		self.area = m.pi * ((radius + dr)**2 - radius**2)
		super().__init__(
			mean = t_acc,
			amplitude = self.compute_norm(dr = dr),
			std = self.sigma_time
		)
		# print(self.radius, self.amplitude)

	def compute_norm(self, dr = 0.1):
		if inputs.GSE_SURFACE_DENSITY_MODE == "exponential":
			# Exponentially declining surface density
			scale_radius = inputs.GSE_SCALE_RADIUS # base value: 2 kpc
			# scale_radius = 2. # kpc
			# scale_radius = 2.5 # kpc
			# scale_radius = 4 # kpc
			norm = self.mgas / (scale_radius**2 * self.sigma_time)
			norm /= (2 * m.pi)**(3/2)
			norm *= m.exp(-self.radius / scale_radius)
			norm *= 1.0e-9
			norm /= 1 + m.exp((self.radius - 17) / 1)
			return norm
		elif inputs.GSE_SURFACE_DENSITY_MODE == "constant":
			# Constant surface density
			norm = self.mgas / (MAX_SF_RADIUS**2 * self.sigma_time)
			norm /= m.sqrt(2) * m.pi**(3/2)
			norm *= 1.0e-9
			norm /= 1 + m.exp((self.radius - 17) / 1)
			return norm
		else:
			raise ValueError("Bruh")

		# Gaussian -> may need debugged?
		# mass_per_length = m.exp(-(self.radius - self.r_acc)**2 / (
		# 	2 * self.sigma_radius**2))
		# mass_per_length *= self.mgas / (self.sigma_radius * m.sqrt(2 * m.pi))
		# mass_at_this_radius = mass_per_length * dr
		# mass_acc_rate_norm = mass_at_this_radius / (
		# 	self.sigma_time * m.sqrt(2 * m.pi))
		# mass_acc_rate_norm *= 1.0e-9 # Msun/Gyr -> Msun/yr
		# return mass_acc_rate_norm / self.area


class expifr_with_GSE(expifr, GSE):

	def __init__(self, radius, dt = 0.01, dr = 0.1, **kwargs):
		expifr.__init__(self, radius, dt = dt, dr = dr)
		GSE.__init__(self, radius, dr = dr, **kwargs)

	def __call__(self, time):
		return expifr.__call__(self, time) + GSE.__call__(self, time)


class beta_phi_in_GSE:

	def __init__(self, radius, dr = 0.1, dt = 0.01, **kwargs):
		self.sigmadotin_cgm = expifr(radius, dr = dr, dt = dt)
		self.sigmadotin_gse = GSE(radius, dr = dr, **kwargs)
		self.mw_beta_phi_in = inputs.RADIAL_GAS_FLOW_BETA_PHI_IN
		self.gse_beta_phi_in = inputs.RADIAL_GAS_FLOW_GSE_BETA_PHI_IN
		# print(self.mw_beta_phi_in(8, 5))

	def __call__(self, radius, time):
		if callable(self.mw_beta_phi_in):
			mw_beta_phi_in = self.mw_beta_phi_in(radius, time)
		else:
			mw_beta_phi_in = self.mw_beta_phi_in
		if callable(self.gse_beta_phi_in):
			gse_beta_phi_in = self.gse_beta_phi_in(radius, time)
		else:
			gse_beta_phi_in = self.gse_beta_phi_in
		sigmadotin_cgm = self.sigmadotin_cgm(time)
		sigmadotin_gse = self.sigmadotin_gse(time)
		return (sigmadotin_cgm * mw_beta_phi_in +
			sigmadotin_gse * gse_beta_phi_in) / (
			sigmadotin_cgm + sigmadotin_gse)



class Zin_CGM(modified_exponential):

	def __init__(self, elem):
		super().__init__(
			norm = vice.solar_z[elem] * 10**inputs.CGM_FINAL_METALLICITY,
			rise = inputs.CGM_METALLICITY_GROWTH_TIMESCALE,
			timescale = float("inf"))
		self.elem = elem


# class Zin_with_GSE(Zin_acc, GSE, expifr):
class Zin_with_GSE:

	def __init__(self, radius, elem, dr = 0.1, dt = 0.01, **kwargs):
		self.sigmadotin_cgm = expifr(radius, dr = dr, dt = dt)
		self.sigmadotin_gse = GSE(radius, dr = dr, **kwargs)
		self.zin_cgm = Zin_CGM(elem)
		self.zin_gse = {
			"fe": self.sigmadotin_gse.fe_h,
			"o": self.sigmadotin_gse.o_h,
			"mg": self.sigmadotin_gse.mg_h
		}[elem.lower()]
		self.zin_gse = vice.solar_z[elem] * 10**self.zin_gse

	def __call__(self, time):
		sigmadotin_cgm = self.sigmadotin_cgm(time)
		sigmadotin_gse = self.sigmadotin_gse(time)
		zin_cgm = self.zin_cgm(time)
		zin_gse = self.zin_gse
		return (sigmadotin_cgm * zin_cgm + sigmadotin_gse * zin_gse) / (
			sigmadotin_cgm + sigmadotin_gse)
		# print(result)
		# return result

	# def __init__(self, radius, elem, **kwargs):
	# 	Zin_acc.__init__(self, elem)
	# 	GSE.__init__(self, radius, **kwargs)
	# 	expifr.__init__
	# 	# self.acc_rate_func = acc_rate_func

	# def __call__(self, time):
	# 	mdot_acc = acc_rate_func(time)
	# 	mdot_gse = GSE.__call__(time)
	# 	zin_acc = Zin_acc.__call__(time)
	# 	zin_gse = {"fe": self.fe_h, "o": self.o_h, "mg": self.mg_h}[self.elem]
	# 	return (mdot_gse * zin_gse + mdot_acc * zin_acc) / (mdot_gse + mdot_acc)




