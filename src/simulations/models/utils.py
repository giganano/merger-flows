r"""
Utilities for the Johnson et al. (2021) models.

Contents
--------
get_bin_number : <function>
	Obtain the bin number for a given value in an array of bin edges.
interpolate : <function>
	Extrapolate the value of a y-coordinate given two points (x1, y1), (x2, y2)
	and an x-coordinate.
constant : object
	A callable object which always returns a specified constant.
exponential : object
	Exponential growth/decay.
modified_exponential : object
	Exponential growth/decay with a limited exponential growth prefactor.
gaussian : object
	A Gaussian in an arbitrary x-coordinate.
"""

import math as m
import numbers


def get_bin_number(bins, value):
	r"""
	Obtain the bin number of a given value in an array-like object of bin
	edges.

	Parameters
	----------
	bins : array-like
		The bin-edges themselves. Assumed to be sorted from least to greatest.
	value : real number
		The value to obtain the bin number for.

	Returns
	-------
	x : int
		The bin number of ``value`` in the array ``bins``. -1 if the value
		lies outside the extent of the bins.
	"""
	for i in range(len(bins) - 1):
		if bins[i] <= value <= bins[i + 1]: return i
	return -1


def interpolate(x1, y1, x2, y2, x):
	r"""
	Extrapolate a y-coordinate for a given x-coordinate from a line defined
	by two points (x1, y1) and (x2, y2) in arbitrary units.

	Parameters
	----------
	x1 : real number
		The x-coordinate of the first point.
	y1 : real number
		The y-coordinate of the first point.
	x2 : real number
		The x-coordinate of the second point.
	y2 : real number
		The y-coordinate of the second point.
	x : real number
		The x-coordinate to extrapolate a y-coordinate for.

	Returns
	-------
	y : real number
		The y-coordinate such that the point (x, y) lies on the line defined
		by the points (x1, y1) and (x2, y2).
	"""
	return (y2 - y1) / (x2 - x1) * (x - x1) + y1


def choose(a, b):
	r"""
	Evaluate the mathematical operation a choose b.
	"""
	numerator = 1
	denominator = 1
	while b:
		numerator *= a
		denominator *= b
		a -= 1
		b -= 1
	return numerator / denominator


class constant:

	r"""
	Generic constant with a generic x-coordinate.

	Parameters
	----------
	amplitude : real number [default : 1]
		The attribute ``amplitude``, initialize via keyword argument. See
		below.

	Attributes
	----------
	amplitude : real number [default : 1]
		The value of the constant in arbitrary units. When this object is
		called with the value of some x-coordinate, this value will be
		returned.

	Calling
	-------
	This object can be called with any arbitrary x-coordinate, and the
	value of the attribute ``amplitude`` will be returned always.
	"""

	def __init__(self, amplitude = 1):
		self.amplitude = amplitude

	def __call__(self, x):
		return self.amplitude

	@property
	def amplitude(self):
		r"""
		Type : real number

		Default : 1

		The value of the constant at all x-coordinates, in arbitrary units.
		"""
		return self._amplitude

	@amplitude.setter
	def amplitude(self, value):
		if isinstance(value, numbers.Number):
			self._amplitude = value
		else:
			raise TypeError("Amplitude must be a real number. Got: %s" % (
				type(value)))


class sinusoid:

	def __init__(self, amplitude = 1, period = 1, phase = 0):
		self.amplitude = amplitude
		self.period = period
		self.phase = phase

	def __call__(self, x):
		return self.amplitude * m.sin(2 * m.pi * (x - self.phase) / self.period)


class tilted_sinusoid(sinusoid):
    
    def __init__(self, amplitude = 1, period = 1, phase = 0, skewness = 1):
        super().__init__(amplitude = amplitude, period = period, phase = phase)
        self.skewness = skewness
    
    def __call__(self, x):
        s = 0
        for i in range(1, self.skewness + 1):
            a = 2 * choose(2 * self.skewness, self.skewness - i)
            a /= choose(2 * self.skewness, self.skewness)
            a *= super().__call__(i * x) / i
            # a = super().__call__(i * x) / i
            s += a
        return s


class exponential:

	r"""
	Generic exponential growth/decay with time.

	Parameters
	----------
	**kwargs : real numbers
		The attributes ``norm`` and ``timescale`` can be initialized via
		keyword arguments to this function. See below.

	Attributes
	----------
	norm : real number [default : 1]
		The value of the exponential function at time = 0.
	timescale : real number [default : 1]
		The e-folding timescale of the exponential. If positive, this object
		describes exponential decay. If negative, it describes exponential
		growth.

	Calling
	-------
	This object can be called with only the time coordinate in the same units
	as the attribute ``timescale``. The result will be in the units of the
	attribute ``norm``.
	"""

	def __init__(self, norm = 1, timescale = 1):
		self.norm = norm
		self.timescale = timescale

	def __call__(self, time):
		return self.norm * m.exp(-time / self.timescale)

	@property
	def norm(self):
		r"""
		Type : real number

		Default : 1

		The normalization of the exponential (i.e. value at time = 0).
		"""
		return self._norm

	@norm.setter
	def norm(self, value):
		if isinstance(value, numbers.Number):
			self._norm = float(value)
		else:
			raise TypeError("Normalization must be real number. Got: %s" % (
				type(value)))

	@property
	def timescale(self):
		r"""
		Type : real number
		
		Default : 1

		The e-folding timescale of the exponential. If positive, this object
		describes exponential decay. If negative, it describes exponential
		growth.
		"""
		return self._timescale

	@timescale.setter
	def timescale(self, value):
		if isinstance(value, numbers.Number):
			if value:
				self._timescale = float(value)
			else:
				raise ValueError("Timescale must be nonzero.")
		else:
			raise TypeError("Timescale must be a real number. Got: %s" % (
				type(value)))


class logistic(exponential):

	def __init__(self, midpoint = 0, scale = 1, minimum = 0, maximum = 1):
		super().__init__(norm = 1, timescale = scale)
		self.midpoint = midpoint
		self.minimum = minimum
		self.maximum = maximum

	def __call__(self, x):
		return self.minimum + (self.maximum - self.minimum) / (
			1 + super().__call__(x - self.midpoint))


class modified_exponential(exponential):

	r"""
	The modified exponential evolution employed by the fiducial inside-out
	models in Johnson et al. (2021), defined in the following manner:

	.. math:: f(t) \sim (1 - e^{-t/\tau_\text{rise}})e^{-t/\tau}

	Inherits from ``exponential``.

	Parameters
	----------
	**kwargs : real numbers
		All attributes can be assigned via keyword arguments on initialization.

	Attributes
	----------
	rise : real number [default : 1]
		The rise time :math:`\tau_\text{rise}` of the modified exponential.

	Other attributes are inherited from the ``exponential`` object.

	Calling
	-------
	This object can be called with only the time coordinate in the same units
	as the attribute ``timescale``. The result will be in the units of the
	attribute ``norm``.
	"""

	def __init__(self, norm = 1, timescale = 1, rise = 1):
		super().__init__(norm = norm, timescale = timescale)
		self.rise = rise

	def __call__(self, time):
		return (1 - m.exp(-time / self.rise)) * super().__call__(time)

	@property
	def rise(self):
		r"""
		Type : real number

		Default : 1

		The rise time of the modified exponential. Positive definite.
		"""
		return self._rise

	@rise.setter
	def rise(self, value):
		if isinstance(value, numbers.Number):
			if value > 0:
				self._rise = float(value)
			else:
				raise ValueError("Rise must be positive definite. Got: %g" % (
					value))
		else:
			raise TypeError("Rise must be a numerical value. Got: %s" % (
				type(value)))


class gaussian:

	r"""
	A function describing a Gaussian function in some arbitrary x-coordinate.

	Parameters
	----------
	**kwargs : real numbers
		All attributes can be initialized via keyword arguments. See below.

	Attributes
	----------
	mean : real number
		The value of the x-coordinate at which the Gaussian is at its peak.
	amplitude : real number
		The value of the Gaussian at the peak.
	std : real number
		The standard deviation of the Gaussian in the same units as the
		x-coordinate.

	Calling
	-------
	This object can be called with only the value of the x-coordinate in the
	same units as the attribute ``std``. The return value will have the same
	units as the attribute ``amplitude``.
	"""

	def __init__(self, mean = 0, amplitude = 1, std = 1):
		self.mean = mean
		self.amplitude = amplitude
		self.std = std

	def __call__(self, x):
		return self.amplitude * m.exp(-(x - self.mean)**2 / (2 * self.std**2))

	@property
	def mean(self):
		r"""
		Type : real number

		Default : 0

		The value of the x-coordinate in arbitrary units at which the
		Gaussian has its peak.
		"""
		return self._mean

	@mean.setter
	def mean(self, value):
		if isinstance(value, numbers.Number):
			self._mean = float(value)
		else:
			raise TypeError("Mean must be a real number. Got: %s" % (
				type(value)))

	@property
	def amplitude(self):
		r"""
		Type : real number

		Default : 1

		The amplitude of the Gaussian at its peak, in arbitrary units.
		"""
		return self._amplitude

	@amplitude.setter
	def amplitude(self, value):
		if isinstance(value, numbers.Number):
			self._amplitude = float(value)
		else:
			raise TypeError("Amplitude must be a real number. Got: %s" % (
				type(value)))

	@property
	def std(self):
		r"""
		Type : real number

		Default : 1

		The standard deviation of the Gaussian, in arbitrary units.
		"""
		return self._std

	@std.setter
	def std(self, value):
		if isinstance(value, numbers.Number):
			if value:
				self._std = float(value)
			else:
				raise ValueError("Std must be nonzero.")
		else:
			raise TypeError("Std must be a real number. Got: %s" % (
				type(value)))

