
from vice.toolkit import J21_sf_law
from .models.utils import logistic
from . import inputs

class sfe(J21_sf_law):

	# _CRITICAL_SURFACE_DENSITY_ = 1e8 # Msun/kpc^2
	_CRITICAL_SURFACE_DENSITY_ = 1e6 # Msun/kpc^2

	_KS_PLAW_INDEX_ = 1.5 # Kennicutt-Schmidt power-law index

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if inputs.HALO_SIMMER_TO_BOIL:
			self.disk_collapse = simmer_to_boil()
		else:
			pass

	def __call__(self, time, arg2):
		# return 2
		molecular = self.molecular(time)
		# molecular = 2
		if self.mode in ["ifr", "gas"]:
			# arg2 represents the gas supply in Msun
			sigma_gas = arg2 / self.area
			if sigma_gas < 1.0e-12: return 1.e+12 # avoid ZeroDivisionError
			if sigma_gas >= self._CRITICAL_SURFACE_DENSITY_:
				result = molecular
			else:
				# print("b")
				# print(molecular)
				# print(sigma_gas)
				# print(self._CRITICAL_SURFACE_DENSITY_)
				# print(self._KS_PLAW_INDEX_)
				result = molecular * (sigma_gas /
						self._CRITICAL_SURFACE_DENSITY_)**-self._KS_PLAW_INDEX_
		else:
			# arg2 represents the star formation rate in Msun/yr
			sigma_sfr = arg2 / self.area
			sigma_sfr *= 1e9 # yr^-1 -> Gyr^-1
			if sigma_sfr <= 0: return 1.e-12 # avoid ZeroDivisionError
			scaling = (sigma_sfr * molecular /
				self._CRITICAL_SURFACE_DENSITY_)**(1 / self._KS_PLAW_INDEX_ - 1)
			# return molecular * max(1, scaling)
			result = molecular * scaling
		if inputs.HALO_SIMMER_TO_BOIL: result *= (1 + self.disk_collapse(time))
		return result



class simmer_to_boil(logistic):

	def __init__(self,
		midpoint = inputs.HALO_SFE_MIDPOINT_TIME,
		scale = inputs.HALO_SFE_DECAY_TIMESCALE,
		minimum = inputs.HALO_SFE_PREFACTOR,
		maximum = 0):
		super().__init__(midpoint = midpoint, scale = scale,
			minimum = minimum, maximum = maximum)

