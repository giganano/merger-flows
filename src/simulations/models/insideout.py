r"""
This file declares the time-dependence of the star formation history at a
given radius in the fiducial inside-out model from Johnson et al. (2021).
"""

from scipy.optimize import bisect
from ..._globals import END_TIME
from .utils import modified_exponential, get_bin_number, interpolate
from .normalize import normalize
from .gradient import gradient
import math as m
import os

_TAU_RISE_ = 2.0 # Gyr -- left in b/c otherwise lateburst.py can't import
TAUSFHMAX = 200
TAURISEMAX = 2 * END_TIME

# r_rise = 6.5
# r_sfh = 3.26

def exponential_tau_rise(radius):
	return 2 * m.exp(radius / 6.5)


def exponential_tau_sfh(radius):
	return 2 + 2 * m.exp(radius / 4.7)



class insideout(modified_exponential):

	r"""
	The inside-out SFH model from Johnson et al. (2021).

	Parameters
	----------
	radius : float
		The galactocentric radius in kpc of a given annulus in the model.
	dt : float [default : 0.01]
		The timestep size of the model in Gyr.
	dr : float [default : 0.1]
		The width of the annulus in kpc.

	Functions
	---------
	- timescale [staticmethod]

	Other atributes and functionality are inherited from
	``modified_exponential`` declared in ``src/simulations/models/utils.py``.
	"""

	# def __init__(self, radius, dt = 0.01, dr = 0.1):
	# 	# super().__init__(timescale = insideout.timescale(radius),
	# 	# 	rise = _TAU_RISE_)
	# 	# self.norm *= normalize(self, gradient, radius, dt = dt, dr = dr)
	# 	tausfh, taurise = find_tausfh_taurise(radius)
	# 	if m.isnan(tausfh): tausfh = TAUSFHMAX
	# 	if m.isnan(taurise): taurise = TAURISEMAX
	# 	super().__init__(timescale = tausfh, rise = taurise)
	# 	self.norm *= normalize(self, gradient, radius, dt = dt, dr = dr)

	# def __call__(self, time):
	# 	return super().__call__(time)

	def __init__(self, radius, dt = 0.01, dr = 0.1):
		super().__init__(
			timescale = exponential_tau_sfh(radius),
			rise = exponential_tau_rise(radius)
		)
		self.norm *= normalize(self, gradient, radius, dt = dt, dr = dr)

	@staticmethod
	def timescale(radius, Re = 5):
		r"""
		Determine the timescale of star formation at a given radius reported
		by Sanchez (2020) [1]_.

		Parameters
		----------
		radius : real number
			Galactocentric radius in kpc.
		Re : real number [default : 5]
			The effective radius (i.e. half-mass radius) of the galaxy in kpc.

		Returns
		-------
		tau_sfh : real number
			The e-folding timescale of the star formation history at that
			radius. The detailed time-dependence on the star formation history
			has the following form:

			.. math:: \dot{M}_\star \sim
				(1 - e^{-t / \tau_\text{rise}})e^{-t / \tau_\text{sfh}}

			where :math:`\tau_\text{rise}` = 2 Gyr and :math:`\tau_\text{sfh}`
			is the value returned by this function.

		.. [1] Sanchez (2020), ARA&A, 58, 99
		"""
		radius /= Re # convert to units of Re
		radii, timescales = _read_sanchez_data()
		idx = get_bin_number(radii, radius)
		if idx != -1:
			return interpolate(radii[idx], timescales[idx], radii[idx + 1],
				timescales[idx + 1], radius)
		else:
			return interpolate(radii[-2], timescales[-2], radii[-1],
				timescales[-1], radius)


def observed_age_gradient(radius, slope = -0.375, value_at_rsun = 5.21, rsun = 8):
	return value_at_rsun + slope * (radius - rsun)


class driver(modified_exponential):

	# Changes the function call to return the integral of the SFH

	def __init__(self, radius, **kwargs):
		self.radius = radius
		super().__init__(**kwargs)

	def __call__(self, time):
		term1 = m.exp(-time * (1 / self.rise + 1 / self.timescale)) - 1
		term1 /= 1 / self.rise + 1 / self.timescale
		term2 = m.exp(-time / self.timescale) - 1
		term2 *= self.timescale
		return term1 - term2

class find_tausfh_driver(driver):

	def __call__(self, tausfh):
		target = END_TIME - observed_age_gradient(self.radius) # age -> time
		self.timescale = tausfh
		return super().__call__(target) - 0.5 * super().__call__(END_TIME)

class find_taurise_driver(driver):

	def __call__(self, taurise):
		target = END_TIME - observed_age_gradient(self.radius) # age -> time
		self.rise = taurise
		return super().__call__(target) - 0.5 * super().__call__(END_TIME)

def find_tausfh_taurise(radius):
	taurise = 2
	driver = find_tausfh_driver(radius, rise = taurise)
	if driver(0.1) * driver(TAUSFHMAX) < 0:
		tausfh = bisect(driver, 0.1, TAUSFHMAX)
		return [tausfh, taurise]
	else: pass
	driver = find_taurise_driver(radius, timescale = TAUSFHMAX)
	if driver(0.1) * driver(TAURISEMAX) < 0:
		taurise = bisect(driver, 0.1, TAURISEMAX)
		return [TAUSFHMAX, taurise]
	else:
		return [float("nan"), float("nan")]


def _read_sanchez_data():
	r"""
	Reads the Sanchez (2020) [1]_ star formation timescale data.

	Returns
	-------
	radii : list
		Radius in units of the effective radius :math:`R_e` (i.e. the
		half-mass radius).
	timescales : list
		The star formation timescales in Gyr associated with each effective
		radius.

	.. [1] Sanchez (2020), ARA&A, 58, 99
	"""
	radii = []
	timescales = []

	# This function likely won't be called from this directory -> full path
	with open("%s/sanchez_tau_sfh.dat" % (
		os.path.abspath(os.path.dirname(__file__))), 'r') as f:

		line = f.readline()
		
		# read past the header
		while line[0] == '#':
			line = f.readline()

		# pull in each line until end of file is reached
		while line != "":
			line = [float(i) for i in line.split()]
			radii.append(line[0])
			timescales.append(line[1])
			line = f.readline()

		f.close()
	return [radii, timescales]

