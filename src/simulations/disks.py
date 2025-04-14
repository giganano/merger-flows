r"""
The diskmodel objects employed in the Johnson et al. (2021) study.
"""

try:
	ModuleNotFoundError
except NameError:
	ModuleNotFoundError = ImportError
try:
	import vice
except (ModuleNotFoundError, ImportError):
	raise ModuleNotFoundError("Could not import VICE.")
if vice.version[:2] < (1, 2):
	raise RuntimeError("""VICE version >= 1.2.0 is required to produce \
Johnson et al. (2021) figures. Current: %s""" % (vice.__version__))
else: pass
from vice.yields.presets import JW20
from vice.toolkit import hydrodisk
vice.yields.sneia.settings['fe'] *= 10**0.1
from .._globals import END_TIME, MAX_SF_RADIUS, ZONE_WIDTH
from . import gasflows
from . import outflows
from . import migration
from . import models
from .models.normalize import normalize_ifrmode
from .models.gradient import gradient
from .models import mergers
from . import inputs
from . import sfe
from .models.utils import get_bin_number, interpolate, modified_exponential
from .models.gradient import gradient
# import warnings
import math as m
import sys



class diskmodel(vice.milkyway):

	r"""
	A milkyway object tuned to the Johnson et al. (2021) models specifically.

	Parameters
	----------
	zone_width : ``float`` [default : 0.1]
		The width of each annulus in kpc.
	name : ``str`` [default : "diskmodel"]
		The name of the model; the output will be stored in a directory under
		this name with a ".vice" extension.
	spec : ``str`` [default : "static"]
		A keyword denoting the time-dependence of the star formation history.
		Allowed values:

		- "static"
		- "insideout"
		- "lateburst"
		- "outerburst"

	verbose : ``bool`` [default : True]
		Whether or not the run the models with verbose output.
	migration_mode : ``str`` [default : "diffusion"]
		A keyword denoting the time-dependence of stellar migration.
		Allowed values:

		- "diffusion"
		- "linear"
		- "sudden"
		- "post-process"

	kwargs : varying types
		Other keyword arguments to pass ``vice.milkyway``.

	Attributes and functionality are inherited from ``vice.milkyway``.
	"""

	def __init__(self, zone_width = 0.1, timestep_size = 0.01,
		name = "diskmodel", spec = "static", verbose = True,
		migration_mode = "diffusion", elements = ["fe", "o"], **kwargs):
		super().__init__(zone_width = zone_width, name = name,
			verbose = verbose, **kwargs)
		if self.zone_width <= 0.2 and self.dt <= 0.02 and self.n_stars >= 6:
			Nstars = 3102519
		else:
			Nstars = 2 * int(MAX_SF_RADIUS / zone_width * END_TIME / self.dt *
				self.n_stars)
		self.migration.stars = migration.gaussian_migration(self.annuli,
			zone_width = zone_width,
			filename = "%s_analogdata.out" % (self.name),
			post_process = self.simple)
		# self.evolution = star_formation_history(spec = spec,
		# 	zone_width = zone_width)
		kwargs = {
			"spec": spec,
			"zone_width": zone_width
		}
		if spec in ["expifr", "expifr_gse"]:
			self.mode = "ifr"
			self.evolution = accretion_history(**kwargs)
		else:
			self.mode = "sfr"
			self.evolution = star_formation_history(**kwargs)
		self.dt = timestep_size
		self.elements = elements

		for i in range(self.n_zones):
			if inputs.OUTFLOWS in ["empirical_calib", "J25", "rc25_constant"]:
				cls = {
					"empirical_calib": outflows.empirical_calib,
					"J25": outflows.J25,
					"rc25_constant": outflows.rc25_constant,
				}[inputs.OUTFLOWS]
				kwargs = {
					"timestep": self.zones[i].dt
				}
				if i: kwargs["evol"] = self.zones[0].eta.evol
				self.zones[i].eta = cls(self,
					i * zone_width + 1.e-6, **kwargs)
			elif inputs.OUTFLOWS == "constant_t_and_r":
				self.zones[i].eta = outflows.constant_t_and_r(
					inputs.OUTFLOWS_CONST_ETA)
			elif inputs.OUTFLOWS is None:
				self.zones[i].eta = 0
			else:
				raise ValueError("Bad outflow setting in input file.")

		for i in range(self.n_zones):
			area = m.pi * ZONE_WIDTH**2 * ((i + 1)**2 - i**2)
			self.zones[i].Mg0 = 0
			self.zones[i].tau_star = sfe.sfe(area, mode = self.mode)
			self.zones[i].RIa = "exp"
			self.zones[i].tau_ia = 1.5
			# if self.mode == "ifr" and zone_width * (i + 0.5) > MAX_SF_RADIUS:
			# 	self.zones[i].tau_star = float("inf")
			# else:
			# 	self.zones[i].tau_star = sfe.sfe(area, mode = self.mode)

		# setup radial gas flow
		if inputs.RADIAL_GAS_FLOWS is not None:
			kwargs = {
				"onset": inputs.RADIAL_GAS_FLOW_ONSET,
				"dr": zone_width,
				"dt": self.dt
			}
			if self.mode == "ifr":
				self.migration.gas.callback = True
				kwargs["outfilename"] = None
			else:
				kwargs["outfilename"] = "%s_gasvelocities.out" % (self.name)
			callkwargs = {}
			if inputs.RADIAL_GAS_FLOWS == "constant":
				if self.mode == "ifr":
					for i in range(self.n_zones):
						for j in range(self.n_zones):
							if spec == "expifr":
								obj = gasflows.constant_ifrmode
							elif spec == "expifr_gse":
								raise ValueError("Need to set up gas flows.")
							else:
								raise ValueError("Bruh.")
							if abs(i - j) == 1:
								self.migration.gas[i][j] = obj(
									i * zone_width,
									inputs.RADIAL_GAS_FLOW_SPEED,
									inward = i > j,
									**kwargs)
							else: pass
				else:
					self.radialflow = gasflows.constant(
						inputs.RADIAL_GAS_FLOW_SPEED, **kwargs)
			elif inputs.RADIAL_GAS_FLOWS == "oscillatory":
				self.radialflow = gasflows.oscillatory(
					inputs.RADIAL_GAS_FLOW_MEAN,
					inputs.RADIAL_GAS_FLOW_AMPLITUDE,
					inputs.RADIAL_GAS_FLOW_PERIOD,
					**kwargs)
			elif inputs.RADIAL_GAS_FLOWS == "linear":
				self.radialflow = gasflows.linear(
					dvdr = inputs.RADIAL_GAS_FLOW_DVDR,
					**kwargs)
			elif inputs.RADIAL_GAS_FLOWS == "amd_pwd":
				if self.mode == "ifr":
					mstar_container = gasflows.container()
					mstar_container.mstar = 0
					mstar_container.sfrs = self.n_zones * [0.]
					for i in range(self.n_zones):
						for j in range(self.n_zones):
							if spec == "expifr":
								bphiin = inputs.RADIAL_GAS_FLOW_BETA_PHI_IN
							elif spec == "expifr_gse":
								bphiin = mergers.beta_phi_in_GSE(
									zone_width * (i + 0.5),
									dr = zone_width,
									dt = self.dt)
							else: raise ValueError("Bruh")
							bphiout = inputs.RADIAL_GAS_FLOW_BETA_PHI_OUT
							if abs(i - j) == 1:
								self.migration.gas[i][j] = gasflows.amd_pwd_ifrmode(
									i * zone_width,
									self,
									inward = i > j,
									mstar_container = mstar_container,
									gamma = inputs.RADIAL_GAS_FLOW_PWDGAMMA,
									beta_phi_in = bphiin,
									beta_phi_out = bphiout,
									**kwargs)
							else: pass
				else: raise ValueError("Bruh")
			elif inputs.RADIAL_GAS_FLOWS == "angular_momentum_dilution":
				if self.mode == "ifr":
					for i in range(self.n_zones):
						for j in range(self.n_zones):
							if spec == "expifr":
								bphiin = inputs.RADIAL_GAS_FLOW_BETA_PHI_IN
							elif spec == "expifr_gse":
								bphiin = mergers.beta_phi_in_GSE(
									zone_width * (i + 0.5),
									dr = zone_width,
									dt = self.dt)
							else:
								raise ValueError("Bruh.")
							bphiout = inputs.RADIAL_GAS_FLOW_BETA_PHI_OUT
							if abs(i - j) == 1:
								self.migration.gas[i][j] = gasflows.amd_ifrmode(
									i * zone_width,
									self,
									inward = i > j,
									beta_phi_in = bphiin,
									beta_phi_out = bphiout,
									**kwargs)
							else: pass
				else:
					self.radialflow = gasflows.amd_sfrmode(self,
						beta_phi_in = inputs.RADIAL_GAS_FLOW_BETA_PHI_IN,
						beta_phi_out = inputs.RADIAL_GAS_FLOW_BETA_PHI_OUT,
						**kwargs)
			elif inputs.RADIAL_GAS_FLOWS == "potential_well_deepening":
				self.radialflow = gasflows.pwd(self,
					gamma = inputs.RADIAL_GAS_FLOW_PWDGAMMA, **kwargs)
			elif inputs.RADIAL_GAS_FLOWS == "ora":
				self.radialflow = gasflows.ora(self, **kwargs)
				callkwargs["recycling"] = 0.4
			else:
				raise ValueError(
					"Unrecognized radial gas flow setting: %s" % (
						inputs.RADIAL_GAS_FLOWS))

			if self.mode == "ifr":
				self.outfile = open("%s_gasvelocities.out" % (self.name), 'w')
				self.outfile.write("# Time [Gyr]    ")
				self.outfile.write("Radius [kpc]    ")
				self.outfile.write("ISM radial velocity [kpc/Gyr]\n")
				for i in range(self.n_zones):
					for j in range(self.n_zones):
						if isinstance(self.migration.gas[i][j], gasflows.base):
							self.migration.gas[i][j].outfile = self.outfile
						else: pass
			else:
				self.radialflow.setup(self, **callkwargs)

		else:
			pass

		for i in range(self.n_zones):
			self.zones[i].Zin = {}
			for elem in self.zones[i].elements:
				if spec == "expifr_gse":
					self.zones[i].Zin[elem] = mergers.Zin_with_GSE(
						zone_width * (i + 0.5),
						elem,
						dr = zone_width,
						dt = self.dt)
				else:
					self.zones[i].Zin[elem] = mergers.Zin_CGM(elem)


	def run(self, *args, **kwargs):
		out = super().run(*args, **kwargs)
		self.migration.stars.close_file()
		if self.mode == "ifr" and inputs.RADIAL_GAS_FLOWS is not None:
			self.outfile.close()
		else: pass
		return out

	@classmethod
	def from_config(cls, config, **kwargs):
		r"""
		Obtain a ``diskmodel`` object with the parameters encoded into a
		``config`` object.

		**Signature**: diskmodel.from_config(config, **kwargs)

		Parameters
		----------
		config : ``config``
			The ``config`` object with the parameters encoded as attributes.
			See src/simulations/config.py.
		**kwargs : varying types
			Additional keyword arguments to pass to ``diskmodel.__init__``.

		Returns
		-------
		model : ``diskmodel``
			The ``diskmodel`` object with the proper settings.
		"""
		model = cls(zone_width = config.zone_width,
			timestep_size = config.timestep_size, elements = config.elements,
			**kwargs)
		model.n_stars = config.star_particle_density
		model.bins = config.bins
		# model.elements = config.elements
		return model


class evol_spec:

	def __init__(self, spec = "static", zone_width = 0.1):
		self._radii = []
		self._evol = []
		i = 0
		max_radius = 20 # kpc, defined by ``vice.milkyway`` object.
		while (i + 1) * zone_width < max_radius:
			self._radii.append((i + 0.5) * zone_width)
			self._evol.append({
					"oscil":		models.insideout_oscil,
					"static": 		models.static,
					"insideout": 	models.insideout,
					"lateburst": 	models.lateburst,
					"outerburst": 	models.outerburst,
					"expifr": 		models.expifr,
					"expifr_gse": 	models.expifr_with_GSE
				}[spec.lower()]((i + 0.5) * zone_width, dr = zone_width))
			i += 1


class star_formation_history(evol_spec):

	r"""
	The star formation history (SFH) of the model galaxy. This object will be
	used as the ``evolution`` attribute of the ``diskmodel``.

	Parameters
	----------
	spec : ``str`` [default : "static"]
		A keyword denoting the time-dependence of the SFH.
	zone_width : ``float`` [default : 0.1]
		The width of each annulus in kpc.

	Calling
	-------
	- Parameters

		radius : ``float``
			Galactocentric radius in kpc.
		time : ``float``
			Simulation time in Gyr.
	"""

	def __call__(self, radius, time):
		# The milkyway object will always call this with a radius in the
		# self._radii array, but this ensures a continuous function of radius
		if radius > MAX_SF_RADIUS:
			return 0
		else:
			idx = get_bin_number(self._radii, radius)
			if idx != -1:
				result = gradient(radius) * interpolate(self._radii[idx],
					self._evol[idx](time), self._radii[idx + 1],
					self._evol[idx + 1](time), radius)
			else:
				result = gradient(radius) * interpolate(self._radii[-2],
					self._evol[-2](time), self._radii[-1], self._evol[-1](time),
					radius)
			if result < 0: result = 0
			return result


class accretion_history(evol_spec):

	def __call__(self, radius, time):
		if radius > MAX_SF_RADIUS:
			return 0
		else:
			idx = get_bin_number(self._radii, radius)
			if idx != -1:
				result = interpolate(self._radii[idx],
					self._evol[idx](time), self._radii[idx + 1],
					self._evol[idx + 1](time), radius)
			else:
				result = interpolate(self._radii[-2],
					self._evol[-2](time), self._radii[-1], self._evol[-1](time),
					radius)
			if result < 0: result = 0
			return result

