
# from .models.utils import modified_exponential
from .models.utils import logistic
import math as m
import vice


# --------------- OSCILLATING SFR --------------- #
SFROSCIL_AMPLITUDE = 0.5
SFROSCIL_PERIOD = 0.4
SFROSCIL_SKEWNESS = 1
SFROSCIL_PHASE = 0 # perfectly in phase
# SFROSCIL_PHASE = SFROSCIL_PERIOD / 2 # perfectly out of phase



# --------------- YIELDS --------------- #
YIELDSOLAR = 1
FE_CC_FRAC = 0.35
METDEPYIELDS = False





# --------------- OUTFLOWS --------------- #
# OUTFLOWS = "J25" # None to turn them off
OUTFLOWS = None
OUTFLOWS_CONST_ETA = 0.5




# --------------- ACCRETION METALLICITY TIME-DEP --------------- #
CGM_FINAL_METALLICITY = -float("inf") # -inf for zero metallicity accretion
CGM_METALLICITY_GROWTH_TIMESCALE = 3



class logistic_amd(logistic):

	def __call__(self, radius, time):
		return super().__call__(radius)


# --------------- RADIAL GAS FLOWS --------------- #
RADIAL_GAS_FLOWS = "angular_momentum_dilution" # None turns them off
# RADIAL_GAS_FLOWS = "constant"
# RADIAL_GAS_FLOWS = None
RADIAL_GAS_FLOW_ONSET = 1 # Gyr -- radial flow starts 1 Gyr in

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
RADIAL_GAS_FLOW_BETA_PHI_IN = logistic_amd(midpoint = 12.5, scale = 2.5,
	minimum = 0.8, maximum = 1)
# def RADIAL_GAS_FLOW_BETA_PHI_IN(r, t):
	# return 0.3 + 0.4 * (1 - m.exp(-t / 2))
RADIAL_GAS_FLOW_BETA_PHI_OUT = 0

# used when RADIAL_GAS_FLOWS = "potential_well_deepening"
RADIAL_GAS_FLOW_PWDGAMMA = 0.3

# used when RADIAL_GAS_FLOWS = "oscillatory"
RADIAL_GAS_FLOW_MEAN = -0.5
RADIAL_GAS_FLOW_AMPLITUDE = 10
RADIAL_GAS_FLOW_PERIOD = 0.2






vice.yields.ccsne.settings["o"] = YIELDSOLAR * vice.solar_z["o"]
vice.yields.sneia.settings["o"] = 0
vice.yields.ccsne.settings["fe"] = FE_CC_FRAC * YIELDSOLAR * vice.solar_z["fe"]
vice.yields.sneia.settings["fe"] = (1 - FE_CC_FRAC) * YIELDSOLAR * vice.solar_z["fe"]


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

