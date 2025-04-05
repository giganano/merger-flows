
from ..inputs import (SFROSCIL_AMPLITUDE, SFROSCIL_PERIOD, SFROSCIL_SKEWNESS,
	SFROSCIL_PHASE)
from .insideout import insideout
from .normalize import normalize
from .gradient import gradient
# from .utils import sinusoid
from .utils import tilted_sinusoid
import math as m


class insideout_oscil(tilted_sinusoid, insideout):

	def __init__(self, radius, dt = 0.01, dr = 0.1,
		amplitude = SFROSCIL_AMPLITUDE, period = SFROSCIL_PERIOD,
		phase = SFROSCIL_PHASE, skewness = SFROSCIL_SKEWNESS):
		tilted_sinusoid.__init__(self, amplitude = amplitude, period = period,
			phase = phase, skewness = skewness)
		insideout.__init__(self, radius, dt = dt, dr = dr)
		self.norm *= normalize(self, gradient, radius, dt = dt, dr = dr)

	def __call__(self, time):
		return insideout.__call__(self, time) * (1 + tilted_sinusoid.__call__(
			self, time))

