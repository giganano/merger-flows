
from ..._globals import MAX_SF_RADIUS
from .utils import exponential
from .normalize import normalize
from .gradient import gradient
import math as m
import os


class expifr(exponential):

	r"""
	Exponentially declining accretion history with no initial gas supply and
	e-folding timescale in Gyr given by expifr.timescale.

	Basic functions of an exponential decline with time, including the __call__
	function, are inherited from the base class.
	"""

	def __init__(self, radius, dt = 0.01, dr = 0.1):
		super().__init__(
			# norm = 0.3 * m.exp(-radius / 2.5), # sigma_crit = 1e6 Msun/kpc^2
			norm = 0.3 * m.exp(-radius / 2.5), # sigma_crit = 1e8 Msun/kpc^2
			timescale = self.timescale(radius))
		self.radius = radius
		self.norm /= 1 + m.exp((radius - 17) / 1)
		# if self.radius >= MAX_SF_RADIUS - dr: self.norm *= 2
		# self.norm *= normalize(self, gradient, radius, dt = dt, dr = dr)

	def __call__(self, time):
		if self.radius <= MAX_SF_RADIUS:
			return super().__call__(time)
		else:
			return 0

	@staticmethod
	def timescale(radius):
		return 1. * m.exp(radius / 3)

