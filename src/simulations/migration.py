
from .._globals import ZONE_WIDTH, END_TIME
from vice.toolkit import hydrodisk
import math as m
import random


class diskmigration(hydrodisk.hydrodiskstars):

	r"""
	A ``hydrodiskstars`` object which writes extra analog star particle data to
	an output file.

	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	mode : ``str`` [default : "linear"]
		A keyword denoting the time-dependence of stellar migration.
		Allowed values:

		- "diffusion"
		- "linear"
		- "sudden"
		- "post-process"

	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.

	Attributes
	----------
	write : ``bool`` [default : False] 	
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.
	"""

	def __init__(self, radbins, mode = "diffusion", filename = "stars.out",
		**kwargs):
		super().__init__(radbins, mode = mode, **kwargs)
		if isinstance(filename, str):
			self._file = open(filename, 'w')
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# use only disk stars in these simulations
		self.decomp_filter([1, 2])

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		if tform == time:
			super().__call__(zone, tform, time) # reset analog star particle
			if self.write:
				if self.analog_index == -1:
					# finalz = 100
					finalz = 0
					analog_id = -1
				else:
					finalz = self.analog_data["zfinal"][self.analog_index]
					analog_id = self.analog_data["id"][self.analog_index]
				self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
					analog_id, finalz))
			else: pass
			return zone
		else:
			return super().__call__(zone, tform, time)

	def close_file(self):
		r"""
		Closes the output file - should be called after the multizone model
		simulation runs.
		"""
		self._file.close()

	@property
	def write(self):
		r"""
		Type : bool

		Whether or not to write out to the extra star particle data output
		file. For internal use by the vice.multizone object only.
		"""
		return self._write

	@write.setter
	def write(self, value):
		if isinstance(value, bool):
			self._write = value
		else:
			raise TypeError("Must be a boolean. Got: %s" % (type(value)))


class gaussian_migration:
	r"""
	A class which controls the Gaussian stellar migration scheme.
	
	The total migration distance $\Delta R$ is drawn from a Gaussian whose 
	width is determined by the star's age and birth radius. The final z-height 
	is drawn from a sech^2 distribution and written to an external file.
	
	Parameters
	----------
	radbins : array-like
		The bins in galactocentric radius in kpc corresponding to each annulus.
	zone_width : float [default : 0.1]
		Width of each radial zone in kpc.
	end_time : float [default : 13.2]
		The final simulation timestep in Gyr.
	filename : ``str`` [default : "stars.out"]
		The name of the file to write the extra star particle data to.
	absz_max : float [default : 3]
		Maximum |z|-height above the midplane in kpc. The default corresponds
		to the maximum value in the h277 sample.
	
	Attributes
	----------
	write : ``bool`` [default : False]     
		A boolean describing whether or not to write to an output file when
		the object is called. The ``multizone`` object, and by extension the
		``milkyway`` object, automatically switch this attribute to True at the
		correct time to record extra data.
		
	Calling
	-------
	Returns the star's current zone at the given time.
	zone : int
		Birth zone of the star.
	tform : float
		Formation time of the star in Gyr.
	time : float
		Current simulation time in Gyr.
	"""
	def __init__(self, radbins, zone_width = ZONE_WIDTH, end_time = END_TIME,
			filename = "stars.out", absz_max = 3., post_process = False):
		self.radial_bins = radbins
		self.zone_width = zone_width
		self.end_time = end_time
		self.absz_max = absz_max
		self.post_process = post_process
		# super().__init__(radbins, mode=None, filename=filename, **kwargs)
		if isinstance(filename, str):
			self._file = open(filename, 'w')
			# Same format as above for compatibility
			self._file.write("# zone_origin\ttime_origin\tanalog_id\tzfinal\n")
		else:
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename)))

		# Multizone object automatically swaps this to True in setting up
		# its stellar population zone histories
		self.write = False

	def __call__(self, zone, tform, time):
		Rform = self.zone_width * (zone + 0.5)
		age = self.end_time - tform
		if tform == time:
			if age > 0:
				# Randomly draw migration distance dR based on age & Rform
				while True: # ensure 0 < Rfinal <= Rmax
					dR = random.gauss(mu=0., sigma=self.migr_scale(age, Rform))
					Rfinal = Rform + dR
					if Rfinal > 0. and Rfinal <= self.radial_bins[-1]:
						break
			else:
				# Stars born in the last simulation timestep won't migrate
				dR = 0.
				Rfinal = Rform
			self.dR = dR
			# Randomly draw final midplane distance and write to file
			if self.write:
				# vertical scale height based on age and Rfinal
				hz = self.scale_height(age, Rfinal)
				# draw from sech^2 distribution
				rng_max = self.sech2_cdf(self.absz_max, hz)
				rng_min = self.sech2_cdf(-self.absz_max, hz)
				rng = random.uniform(rng_min, rng_max)
				finalz = self.inverse_sech2_cdf(rng, hz)
				analog_id = -1
				self._file.write("%d\t%.2f\t%d\t%.2f\n" % (zone, tform,
					analog_id, finalz))
			else:
				pass
			return zone
		else: 
			if self.post_process:
				if time < self.end_time:
					return zone
				else:
					R = Rform + self.dR
					return int((R / self.zone_width))
			else:
				# Interpolate between Rform and Rfinal at current time
				R = self.interpolator(Rform, Rform + self.dR, tform, time)
				# return int((R / self.zone_width) - 0.5) # ??? reason for -0.5?
				return int((R / self.zone_width))

	def close_file(self):
		r"""
		Closes the output file - should be called after the multizone model
		simulation runs.
		"""
		self._file.close()

	@property
	def write(self):
		r"""
		Type : bool

		Whether or not to write out to the extra star particle data output
		file. For internal use by the vice.multizone object only.
		"""
		return self._write

	@write.setter
	def write(self, value):
		if isinstance(value, bool):
			self._write = value
		else:
			raise TypeError("Must be a boolean. Got: %s" % (type(value)))

	def interpolator(self, Rform, Rfinal, tform, time):
		r"""
		Interpolate between the formation and final radius following the fit
		to the h277 data, $\Delta R \propto t^{0.33}$.
		
		Parameters
		----------
		Rform : float
			Radius of formation in kpc.
		Rfinal : float
			Final radius in kpc.
		tform : float
			Formation time in Gyr.
		time : float
			Simulation time in Gyr.
		
		Returns
		-------
		float
			Radius at the current simulation time in kpc.
		"""
		tfrac = (time - tform) / (self.end_time - tform)
		return Rform + (Rfinal - Rform) * (tfrac**0.33)
		# return Rform + (Rfinal - Rform) * (tfrac**0.5)

	@staticmethod
	def migr_scale(age, Rform):
		r"""
		A prescription for $\sigma_{\Delta R}$, the scale of the Gaussian 
		distribution of radial migration.

		$$ \sigma_{\Delta R} = 1.35 (R_\rm{form}/8\,\rm{kpc})^{0.61} 
		(\tau/1\,\rm{Gyr})^{0.33} $$

		Parameters
		----------
		age : float or array-like
			Age of the stellar population in Gyr.
		Rform : float or array-like
			Formation radius of the stellar population in kpc.

		Returns
		-------
		float or array-like
			Scale factor for radial migration $\sigma_{\Delta R}$.
		"""
		return 1.35 * (age ** 0.33) * (Rform / 8) ** 0.61
		# return 1.81 * (age ** 0.33) * (Rform / 8) ** 0.61 # faster
		# return 1.00 * (age ** 0.33) * (Rform / 8) ** 0.61 # slower


		# return 1.82 * (age ** 0.33) * (Rform / 8) ** 0.61
		# return 1.82 * age**0.33
	
	@staticmethod
	def scale_height(age, Rfinal):
		r"""
		The scale height $h_z$ as a function of age and final radius:
		
		$$ h_z = (0.25\,{\rm kpc}) 
		\exp\Big(\frac{\tau-5\,{\rm Gyr}}{7.0\,{\rm Gyr}}\Big) 
		\exp\Big(\frac{R_{\rm final}-8\,{\rm kpc}}{6.0\,{\rm kpc}}\Big) $$

		Parameters
		----------
		age : float or array-like
			Age in Gyr.
		Rfinal : float or array-like
			Final radius $R_{\rm{final}}$ in kpc.

		Returns
		-------
		float
			Scale height $h_z$ in kpc.
		"""
		return 0.25 * m.exp((age - 5.) / 7.) * m.exp((Rfinal - 8.) / 6.)
		# return 0.18 * (age ** 0.63) * (Rform / 8) ** 1.15

	@staticmethod
	def sech2_cdf(z, scale):
		r"""
		The cumulative distribution function (CDF) of the hyperbolic sec-square
		probability distribution function (PDF), which determines the density
		of stars as a function of distance from the midplane $z$. For some
		scaling $h_z$, the PDF is

		$$ \rm{PDF}(z) = \frac{1}{4 h_z} \cosh^{-2}\Big(\frac{z}{2 h_z}\Big) $$

		and the CDF is

		$$ \rm{CDF}(z) = \frac{1}{1 + e^{-z / h_z}} $$

		Parameters
		----------
		x : float
			Independent variable.
		scale : float
			Width of the sech-squared distribution, with the same units as x.

		Returns
		-------
		float
			The value of the CDF at the given x.
		"""
		return 1 / (1 + m.exp(-z / scale))

	@staticmethod
	def inverse_sech2_cdf(cdf, scale):
		r"""
		The inverse of the sech$^2$ CDF.

		For some scaling $h_z$, the $z$ corresponding to the given value of
		the CDF is

		$$ z = -h_z \ln\Big(\frac{1}{\rm{CDF}} - 1\Big) $$

		Parameters
		----------
		cdf : float
			The value of the CDF. Must be within the exclusive interval (0, 1).
		scale : float
			Width of the sech$^2$ distribution. Must be positive.

		Returns
		-------
		float
			The value of $z$ corresponding to the CDF value, in the same units
			as ``scale''.
		"""
		if cdf <= 0. or cdf >= 1.:
			raise ValueError("The value of the CDF must be between 0 and 1.")
		if scale <= 0.:
			raise ValueError("The scale height must be positive.")
		return -scale * m.log(1 / cdf - 1)


class nomigration(gaussian_migration):

	def __call__(self, zone, tform, time):
		return zone

