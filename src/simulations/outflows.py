r"""
Implements an empirically motivated scaling for Galactic outflows.

This prescription assumes that the Galaxy follows the same spatially-resolved
scalings of outflow surface density with the surface density of stellar mass and
star formation observed in strong starburst systems by Reichardt Chu et al.
(2025).
The normalization is set such that the *total* mass outflow rate scales with
the star formation rate and stellar mass in a way that follows Galactic wind
models of the mass-metallicity relation.
"""

from .._globals import END_TIME
from .models.utils import exponential
import math as m

class evoldata:

	def __init__(self, mw_model, timestep = 0.01, recycling = 0.4):
		self.mw_model = mw_model
		self.mstars = self.mw_model.n_zones * [None]
		self.sfrs = self.mw_model.n_zones * [None]
		for i in range(self.mw_model.n_zones):
			self.mstars[i] = [0]
			self.sfrs[i] = [0]
			for j in range(1, int(END_TIME / timestep) + 10):
				sfr = self.mw_model.zones[i].func(j * timestep)
				self.sfrs[i].append(1.e9 * sfr)
				self.mstars[i].append(self.mstars[i][j - 1] + sfr * (
					1 - recycling) * timestep * 1.e9)


class empirical_calib(exponential):

	MZR_NORM = 3.6 # eta at 10^10 Msun
	MZR_PLAW_INDEX = 1 / 3 # eta ~ Mstar^(-1/3)
	MZR_NORM_MSTAR = 1.0e10 # Msun
	ETA_MAX = 100 # occurs well in the dwarf regime for this prescription

	# def __init__(self, mw_model, radius, zone_width = 0.1, timestep = 0.01,
		# recycling = 0.4):
	# def __init__(self, mw_model, radius, rstar = 2.5, gamma_star = 1.5,
	# 	gamma_sfr = 0.2, timestep = 0.1, recycling = 0.4, evol = None):
	def __init__(self, mw_model, radius, reta = 5, rstar = 2.5,
		timestep = 0.1, recycling = 0.4, evol = None):
		super().__init__(norm = 1, timescale = 1)
		self.mw_model = mw_model
		self.radius = radius
		self.rstar = rstar
		# self.gamma_star = gamma_star
		# self.gamma_sfr = gamma_sfr
		self.timestep = timestep
		self.reta = reta
		# self.timescale = self.rstar / (self.gamma_star + self.gamma_sfr)
		self.timescale = self.reta
		if evol is None:
			self.evol = evoldata(mw_model, timestep = timestep,
				recycling = recycling)
		else:
			self.evol = evol

	def __call__(self, time):
		if time > END_TIME: return self.__call__(END_TIME)
		idx = int(time / self.timestep)
		mstar = sum([self.evol.mstars[_][idx] for _ in range(
			self.mw_model.n_zones)])
		
		# Game the exponential object, whose namespace implies evolution with
		# time, into something that evolves with radius.
		if mstar:
			self.norm = self.MZR_NORM
			self.norm *= (mstar / self.MZR_NORM_MSTAR)**(-self.MZR_PLAW_INDEX)
			self.norm *= (self.reta + self.rstar)**2 / self.reta**2
			# self.norm *= (1 + self.gamma_star + self.gamma_sfr)**2
			return min(super().__call__(self.radius), self.ETA_MAX)
		else:
			return self.ETA_MAX


class rc25_constant(empirical_calib):

	MZR_NORM = 2
	MZR_PLAW_INDEX = 0

	# def __call__(self, time):
		# eta = super().__call__(time)
		# if time == 10: print(self.radius, eta)
		# return eta


class constant_t_and_r:

	def __init__(self, value):
		self.value = value

	def __call__(self, time):
		return self.value


class J25(empirical_calib):

	def __init__(self, *args, reta = -7, **kwargs):
		super().__init__(*args, reta = reta, **kwargs)




# class empirical_calib:

# 	MZR_NORM = 3.6 # eta at 10^10 Msun
# 	MZR_PLAW_INDEX = 1 / 3 # eta ~ Mstar^(-1/3)
# 	MZR_NORM_MSTAR = 1.0e10 # Msun
# 	ETA_MAX = 100 # occurs well in the dwarf regime for this prescription

# 	def __init__(self, mw_model, radius, zone_width = 0.1, timestep = 0.01,
# 		recycling = 0.4, evol = None):
# 		self.mw_model = mw_model
# 		self.radius = radius
# 		self.zone_width = zone_width
# 		self.timestep = timestep
# 		self.recycling = recycling
# 		if evol is None:
# 			self.evol = evoldata(mw_model, timestep = timestep,
# 				recycling = recycling)
# 		else:
# 			self.evol = evol


# 	def __call__(self, time):
# 		# grab stellar mass and sfrs in each zone at the current time.
# 		if time > END_TIME: return self.__call__(END_TIME)
# 		idx = int(time / self.timestep)
# 		mstars = [self.evol.mstars[_][idx] for _ in range(self.mw_model.n_zones)]
# 		sfrs = [self.evol.sfrs[_][idx] for _ in range(self.mw_model.n_zones)]

# 		term1 = 0
# 		for i in range(self.mw_model.n_zones):
# 			area = m.pi * (((i + 1) * self.zone_width)**2 -
# 				(i * self.zone_width)**2)
# 			x = (mstars[i] / area)**1.5
# 			x *= (sfrs[i] / area)**1.2
# 			x *= area
# 			term1 += x
# 		if term1:
# 			term1 = sum(sfrs) / term1
# 		else:
# 			return self.ETA_MAX

# 		term2 = (sum(mstars) / self.MZR_NORM_MSTAR)**(-self.MZR_PLAW_INDEX)

# 		eta0 = self.MZR_NORM * term1 * term2
# 		zone = int(self.radius / self.zone_width)
# 		area = m.pi * ((self.radius + self.zone_width)**2 - self.radius**2)
# 		sigma_sfr = sfrs[zone] / area
# 		sigma_star = mstars[zone] / area
# 		eta = eta0 * sigma_star**1.5 * sigma_sfr**0.2
# 		return min(eta, self.ETA_MAX)

