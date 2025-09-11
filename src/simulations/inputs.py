
# from .models.utils import modified_exponential
from .models.utils import logistic
# from .gasflows import logistic_betaphiin
import math as m
import vice
import os

# --------------- GSE PARAMETERS --------------- #
GSE_MASS_RATIO = 1 / 3
# GSE_REFERENCE_MODEL = "%s/../../outputs/expifr/amd-pwd/base" % (
# 	os.path.dirname(os.path.abspath(__file__)))
# GSE_REFERENCE_MODEL = "%s/../../outputs/expifr/amd-pwd/base-highsigmacrit" % (
# 	os.path.dirname(os.path.abspath(__file__)))
# GSE_REFERENCE_MODEL = "%s/../../outputs/expifr/amd/base" % (
# 	os.path.dirname(os.path.abspath(__file__)))
GSE_REFERENCE_MODEL = "%s/../../outputs/expifr/amd/base-highsigmacrit" % (
	os.path.dirname(os.path.abspath(__file__)))
GSE_T_ACC = 3.2
GSE_SIGMA_TIME = 0.5
GSE_REFERENCE_TIME = GSE_T_ACC - GSE_SIGMA_TIME
GSE_SURFACE_DENSITY_MODE = "exponential"
GSE_SCALE_RADIUS = 2 # kpc, 2 is base-line value so far


# Approximate GSE final chemistry read off of the best-fit
# figure from my dwarf galaxy archaeology paper
# GSE_FEH = -0.5
# GSE_OH = -0.45
# GSE_MGH = -0.45

# metal-poor GSE
# GSE_FEH = -1.5
# GSE_OH = -1.45
# GSE_MGH = -1.45

# metal-free GSE
GSE_FEH = -float("inf")
GSE_OH = -float("inf")
GSE_MGH = -float("inf")



# --------------- HALO SIMMER-TO-BOIL --------------- #
HALO_SIMMER_TO_BOIL = True
HALO_SFE_PREFACTOR = 20
HALO_SFE_MIDPOINT_TIME = 2
HALO_SFE_DECAY_TIMESCALE = 0.25



# --------------- OSCILLATING SFR --------------- #
SFROSCIL_AMPLITUDE = 0.5
SFROSCIL_PERIOD = 0.4
SFROSCIL_SKEWNESS = 1
SFROSCIL_PHASE = 0 # perfectly in phase
# SFROSCIL_PHASE = SFROSCIL_PERIOD / 2 # perfectly out of phase



# --------------- YIELDS --------------- #
YIELDSOLAR = 1
FE_CC_FRAC = 0.35
# FE_CC_FRAC = 0.25
METDEPYIELDS = False
FE_CC_MODIFIER = 10**-0.05
FE_IA_MODIFIER = 10**-0.05





# --------------- OUTFLOWS --------------- #
# OUTFLOWS = "J25" # None to turn them off
OUTFLOWS = None
OUTFLOWS_CONST_ETA = 0.5




# --------------- ACCRETION METALLICITY TIME-DEP --------------- #
# CGM_FINAL_METALLICITY = -0.7 # -inf for zero metallicity accretion
CGM_FINAL_METALLICITY = -float("inf")
CGM_METALLICITY_GROWTH_TIMESCALE = 3



# --------------- RADIAL GAS FLOWS --------------- #

class logistic_betaphiin(logistic):

	def __call__(self, radius, time):
		return super().__call__(radius) # time-independent but radius-dependent


RADIAL_GAS_FLOWS = "angular_momentum_dilution" # None turns them off
# RADIAL_GAS_FLOWS = "amd_pwd"
# RADIAL_GAS_FLOWS = "constant"
# RADIAL_GAS_FLOWS = None
RADIAL_GAS_FLOW_ONSET = 0.1 # Gyr -- radial flow starts 1 Gyr in

# used when RADIAL_GAS_FLOWS = "constant"
RADIAL_GAS_FLOW_SPEED = -0.5 # km/s
# def RADIAL_GAS_FLOW_SPEED(time):
# 	return -10 * np.exp(-time / 3)

# used when RADIAL_GAS_FLOWS = "linear"
RADIAL_GAS_FLOW_DVDR = -0.05

# used when RADIAL_GAS_FLOWS = "angular_momentum_dilution"
# RADIAL_GAS_FLOW_BETA_PHI_IN = 0.5
# RADIAL_GAS_FLOW_BETA_PHI_IN = 0.6
# RADIAL_GAS_FLOW_BETA_PHI_IN = 0.8
# RADIAL_GAS_FLOW_BETA_PHI_IN = 0.8
# RADIAL_GAS_FLOW_BETA_PHI_IN = logistic_betaphiin(
# 	midpoint = 17.5, scale = 2.5,
# 	minimum = 0.9, maximum = 0.7)
RADIAL_GAS_FLOW_BETA_PHI_IN = logistic_betaphiin( # works in AMD mode
	midpoint = 12.5, scale = 2.5,
	minimum = 0.8, maximum = 1)
# RADIAL_GAS_FLOW_BETA_PHI_IN = logistic_betaphiin( # trying for AMD+PWD mode
# 	midpoint = 12.5, scale = 2.5,
# 	minimum = 0.9, maximum = 1)

# RADIAL_GAS_FLOW_GSE_BETA_PHI_IN = RADIAL_GAS_FLOW_BETA_PHI_IN
RADIAL_GAS_FLOW_GSE_BETA_PHI_IN = +0.4


# def RADIAL_GAS_FLOW_BETA_PHI_IN(r, t):
	# return 0.3 + 0.4 * (1 - m.exp(-t / 2))
RADIAL_GAS_FLOW_BETA_PHI_OUT = 0

# used when RADIAL_GAS_FLOWS = "potential_well_deepening"
RADIAL_GAS_FLOW_PWDGAMMA = 0.2
# RADIAL_GAS_FLOW_PWDGAMMA = 0

# used when RADIAL_GAS_FLOWS = "oscillatory"
RADIAL_GAS_FLOW_MEAN = -0.5
RADIAL_GAS_FLOW_AMPLITUDE = 10
RADIAL_GAS_FLOW_PERIOD = 0.2





if GSE_MASS_RATIO > 0:
	ref = vice.output(GSE_REFERENCE_MODEL)
	mass = 0
	diff = [abs(_ - GSE_REFERENCE_TIME) for _ in
		ref.zones["zone0"].history["time"]]
	idx = diff.index(min(diff))
	for i in range(len(ref.zones.keys())):
		mass += ref.zones["zone%d" % (i)].history["mgas"][idx]
	GSE_GAS_MASS = mass * GSE_MASS_RATIO
else:
	GSE_GAS_MASS = 0



vice.yields.ccsne.settings["o"] = YIELDSOLAR * vice.solar_z["o"]
vice.yields.sneia.settings["o"] = 0
vice.yields.ccsne.settings["mg"] = YIELDSOLAR * vice.solar_z["mg"]
vice.yields.sneia.settings["mg"] = 0
vice.yields.ccsne.settings["fe"] = FE_CC_FRAC * YIELDSOLAR * vice.solar_z["fe"]
vice.yields.sneia.settings["fe"] = (
	1 - FE_CC_FRAC) * YIELDSOLAR * vice.solar_z["fe"]
vice.yields.ccsne.settings["fe"] *= FE_CC_MODIFIER
vice.yields.sneia.settings["fe"] *= FE_IA_MODIFIER


class metdepyield:

	def __init__(self, baseline, maxincrease = 3, plawindex = -0.5,
		zsun = 0.014):
		self.baseline = baseline
		self.maxincrease = maxincrease
		self.plawindex = plawindex
		self.zsun = zsun

	def __call__(self, z):
		if z:
			prefactor = min((z / self.zsun)**self.plawindex, self.maxincrease)
		else:
			prefactor = self.maxincrease
		return self.baseline * prefactor


if METDEPYIELDS:
	for elem in ["o", "fe"]:
		vice.yields.ccsne.settings[elem] = metdepyield(
			vice.yields.ccsne.settings[elem])
		vice.yields.sneia.settings[elem] = metdepyield(
			vice.yields.sneia.settings[elem])
else: pass

