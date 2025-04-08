r"""
Handles radial gas flows in these models.
"""

from .._globals import MAX_SF_RADIUS, END_TIME, M_STAR_MW
from .models.utils import get_bin_number, sinusoid
from .models.gradient import gradient
from .outflows import evoldata
from vice.toolkit.interpolation import interp_scheme_1d
from vice.milkyway.milkyway import _MAX_RADIUS_ as MAX_RADIUS # 20 kpc
from scipy.integrate import solve_ivp
import numpy as np
import warnings
import vice
from vice import ScienceWarning
import sys


class driver(interp_scheme_1d):

	def __init__(self, *args, dt = 0.01, **kwargs):
		super().__init__(*args, **kwargs)
		self.dt = dt
	
	def __call__(self, x):
		test = super().__call__(x)
		if test < 0:
			return 0
		else:
			return min(test, 0.01 / self.dt - 1.e-9)



class base:

	NORMTIME = 0.01 # Gyr

	def __init__(self, onset = 1, dr = 0.1, dt = 0.01,
		outfilename = "gasvelocities.out"):
		if outfilename is not None:
			self.outfile = open(outfilename, "w")
			self.outfile.write("# Time [Gyr]    ")
			self.outfile.write("Radius [kpc]    ")
			self.outfile.write("ISM radial velocity [kpc/Gyr]\n")
		else:
			self.outfile = None
		self.onset = onset
		self.dr = dr
		self.dt = dt


	def __enter__(self):
		return self


	def __exit__(self, exc_type, exc_value, exc_tb):
		self.outfile.close()
		return exc_value is None


	def write(self, time, radii, velocities):
		if self.outfile is not None:
			for i in range(len(radii)):
				self.outfile.write("%.5e\t%.5e\t%.5e\n" % (
					time, radii[i], velocities[i]))
		else: pass


	def setup(self, mw_model, **kwargs):
		vgas_alltimes = []
		n_zones = int(MAX_RADIUS / self.dr)
		times = [self.dt * i for i in range(int(END_TIME / self.dt) + 10)]
		for i in range(len(times)):
			if i > self.onset / self.dt:
				radii, vgas = self.__call__(i * self.dt, **kwargs)
				vgas_alltimes.append(vgas)
			else:
				radii = [self.dr * i for i in range(n_zones)]
				vgas = len(radii) * [0.]
				vgas_alltimes.append(vgas)
		matrix_elements_inward = []
		matrix_elements_outward = []
		for i in range(n_zones):
			areafracs_inward = []
			areafracs_outward = []
			vgas = [row[i] for row in vgas_alltimes]
			for j in range(len(times)):
				if j >= self.onset / self.dt:
					radius = i * self.dr
					if vgas[j] > 0: # outward flow
						numerator = 2 * (radius + 
							self.dr) * vgas[j] * self.NORMTIME
						numerator -= vgas[j]**2 * self.NORMTIME**2
					else: # inward flow
						numerator = vgas[j]**2 * self.NORMTIME**2
						numerator -= 2 * radius * vgas[j] * self.NORMTIME
					denominator = 2 * radius * self.dr + self.dr**2
					areafrac = numerator / denominator
					if areafrac * self.dt / self.NORMTIME > 1 - 1.e-9:
						warnings.warn("""\
Area fraction larger than 1. Consider comparing results with different \
timestep sizes to assess the impact of numerical artifacts.""", ScienceWarning)
						areafrac = 1 - 1.e-9
					elif areafrac * self.dt / self.NORMTIME < 1.e-9:
						areafrac = 1.e-9
					else: pass
					if vgas[j] > 0:
						areafracs_outward.append(areafrac)
						areafracs_inward.append(1.e-10)
					else:
						areafracs_outward.append(1.e-10)
						areafracs_inward.append(areafrac)
				else:
					areafracs_outward.append(1.e-10)
					areafracs_inward.append(1.e-10)
			matrix_elements_outward.append(
				driver(times, areafracs_outward, dt = self.dt))
			matrix_elements_inward.append(
				driver(times, areafracs_inward, dt = self.dt))
		for i in range(n_zones):
			for j in range(n_zones):
				if i - 1 == j: # inward flows
					mw_model.migration.gas[i][j] = matrix_elements_inward[i]
				elif i + 1 == j: # outward flows
					mw_model.migration.gas[i][j] = matrix_elements_outward[i]
				else:
					mw_model.migration.gas[i][j] = 0


	@staticmethod
	def area_fraction(radius, vgas, dr = 0.1):
		denominator = (radius + dr)**2 - radius**2
		if vgas > 0:
			numerator = (radius + dr)**2 - (
				radius + dr - vgas * base.NORMTIME)**2
		elif vgas < 0:
			numerator = (radius - vgas * base.NORMTIME)**2 - radius**2
		else:
			numerator = 0
		return numerator / denominator


class constant(base):

	def __init__(self, speed, onset = 1, dr = 0.1, dt = 0.01,
		outfilename = "gasvelocities.out"):
		super().__init__(onset = onset, dr = dr, dt = dt,
			outfilename = outfilename)
		self.speed = speed


class constant_ifrmode(constant):

	def __init__(self, radius, *args, inward = True, **kwargs):
		self.radius = radius
		self.inward = inward
		super().__init__(*args, **kwargs)

	def __call__(self, **ism_state):
		if ism_state["time"] < self.onset: return 0
		if self.speed != 0:
			if (self.inward and self.speed < 0) or (
				not self.inward and self.speed > 0):
				self.write(ism_state["time"], [self.radius], [self.speed])
				frac = self.area_fraction(self.radius, self.speed, dr = self.dr)
				if frac < 0:
					frac = 0
				elif frac > 1 - 1.0e-9:
					frac = 1 - 1.0e-9
				else: pass
				return frac
			else:
				return 0
		else:
			# without this seemingly useless if-statement, both inward and
			# outward components write to the output file, resulting in
			# duplicate entries of zero velocity.
			if self.inward: self.write(ism_state["time"], [self.radius], [0])
			return 0


class constant_sfrmode(constant):

	def __call__(self, time, **kwargs):
		radii = [self.dr * i for i in range(int(MAX_RADIUS / self.dr))]
		if callable(self.speed):
			# it's a constant in radius, but not necessarily in time
			speed = self.speed(time, **kwargs)
		else:
			speed = self.speed
		vgas = len(radii) * [speed]
		self.write(time, radii, vgas)
		return [radii, vgas]


class oscillatory(base, sinusoid):

	def __init__(self, average, amplitude, period, phase = 0, onset = 1,
		dr = 0.1, dt = 0.01, outfilename = "gasvelocities.out"):
		base.__init__(self, onset = onset, dr = dr, dt = dt,
			outfilename = outfilename)
		sinusoid.__init__(self, amplitude = amplitude, period = period,
			phase = phase)
		self.average = average

	def __call__(self, time):
		radii = [self.dr * i for i in range(int(MAX_RADIUS / self.dr))]
		vgas = self.average + sinusoid.__call__(self, time)
		vgas = len(radii) * [vgas]
		self.write(time, radii, vgas)
		return [radii, vgas]



class linear(base):

	def __init__(self, dvdr = -0.1, onset = 1, dr = 0.1, dt = 0.01,
		outfilename = "gasvelocities.out"):
		super().__init__(onset = onset, dr = dr, dt = dt,
			outfilename = outfilename)
		self.dvdr = dvdr


	def __call__(self, time):
		radii = [self.dr * i for i in range(int(MAX_RADIUS / self.dr))]
		vgas = [self.dvdr * r for r in radii]
		self.write(time, radii, vgas)
		return [radii, vgas]



class pwd(base):

	def __init__(self, mw_model, gamma = 0.2, onset = 1, dr = 0.1, dt = 0.01,
		recycling = 0.4, outfilename = "gasvelocities.out"):
		super().__init__(onset = onset, dr = dr, dt = dt,
			outfilename = outfilename)
		self.gamma = gamma
		self.mw_model = mw_model
		self.recycling = recycling
		self.evol = evoldata(mw_model, timestep = dt, recycling = recycling)


	def __call__(self, time):
		radii = [self.dr * i for i in range(self.mw_model.n_zones)]
		timestep = int(time / self.dt)
		sfr = 0
		mstar = 0
		for i in range(len(radii)):
			# decrement sfr by 1 - r because the potential well
			# deepening argument is based on the time derivative of the
			# stellar mass, which differs in important detail from the
			# specific star formation rate.
			sfr += (1 - self.recycling) * self.evol.sfrs[i][timestep]
			mstar += self.evol.mstars[i][timestep]
		vgas = [-r * self.gamma * sfr / mstar for r in radii]
		self.write(time, radii, vgas)
		return [radii, vgas]


class amd(base):

	def __init__(self, mw_model, beta_phi_in = 0.7, beta_phi_out = 0, onset = 1,
		dr = 0.1, dt = 0.01, outfilename = "gasvelocities.out"):
		super().__init__(onset = onset, dr = dr, dt = dt,
			outfilename = outfilename)
		self.mw_model = mw_model
		self.beta_phi_in = beta_phi_in
		self.beta_phi_out = beta_phi_out


class amd_ifrmode(amd):

	def __init__(self, radius, *args, inward = True, **kwargs):
		self.radius = radius
		self.inward = inward
		super().__init__(*args, **kwargs)

	def __call__(self, **ism_state):
		if ism_state["time"] < self.onset: return 0
		if callable(self.beta_phi_in):
			beta_phi_in = self.beta_phi_in(self.radius, ism_state["time"])
		else:
			beta_phi_in = self.beta_phi_in
		if callable(self.beta_phi_out):
			beta_phi_out = self.beta_phi_out(self.radius, ism_state["time"])
		else:
			beta_phi_out = self.beta_phi_out
		vgas = ism_state["ofr"] / ism_state["mgas"] * (1 - beta_phi_out)
		vgas -= ism_state["ifr"] / ism_state["mgas"] * (1 - beta_phi_in)
		vgas *= 1e9 # kpc/yr -> kpc/Gyr ~ km/s
		vgas *= self.radius
		if vgas != 0:
			if (self.inward and vgas < 0) or (not self.inward and vgas > 0):
				self.write(ism_state["time"], [self.radius], [vgas])
				frac = self.area_fraction(self.radius, vgas, dr = self.dr)
				if frac < 0:
					frac = 0
				elif frac > 1 - 1.0e-9:
					frac = 1 - 1.0e-9
				else: pass
				return frac
			else:
				return 0
		else:
			# without this seemingly useless if-statement, both inward and
			# outward components write to the output file, resulting in
			# duplicate entries of zero velocity.
			if self.inward: self.write(ism_state["time"], [self.radius], [0])
			return 0

# class amd_ifrmode(amd):

# 	def __call__(self, time):
# 		radii = [self.dr * i for i in range(self.mw_model.n_zones)]
# 		vgas = []
# 		for i in range(len(radii)):
# 			vgas.append(self._velocities[i](time))
# 		self.write(time, radii, vgas)
# 		return [radii, vgas]


# 	def evolve(self, recycling = 0.4):
# 		if callable(self.beta_phi_in):
# 			beta_phi_in = self.beta_phi_in
# 		else:
# 			beta_phi_in = lambda r, t: self.beta_phi_in
# 		if callable(self.beta_phi_out):
# 			beta_phi_out = self.beta_phi_out
# 		else:
# 			beta_phi_out = lambda r, t: self.beta_phi_out
# 		n_zones = self.mw_model.n_zones
# 		evol = []
# 		for i in range(n_zones):
# 			if callable(self.mw_model.zones[i].eta):
# 				eta = self.mw_model.zones[i].eta(time)
# 			else:
# 				eta = self.mw_model.zones[i].eta
# 			if callable(self.mw_model.zones[i].tau_star):
# 				tau_star = self.mw_model.zones[i].tau_star(0,
# 					self.mw_model.zones[i].Mg0)
# 			else:
# 				tau_star = self.mw_model.zones[i].tau_star
# 			sfr = self.mw_model.zones[i].Mg0 / tau_star
# 			ofr = eta * sfr
# 			evol.append({
# 				"time": [0],
# 				"ifr": [self.mw_model.zones[i].func(0)],
# 				"mgas": [self.mw_model.zones[i].Mg0],
# 				"tau_star": [tau_star],
# 				"mstar": [0],
# 				"sfr": [sfr],
# 				"ofr": [ofr],
# 				"vgas": [0]
# 			})

# 		time = 0
# 		while time < END_TIME:
# 			vgas = n_zones * [0.]
# 			mu = n_zones * [0.]
# 			if time >= self.onset:
# 				for i in range(n_zones):
# 					radius = self.dr * (i + 0.5)
# 					# if radius < MAX_SF_RADIUS:
# 					vgas[i] = evol[i]["ofr"][-1] * (
# 						1 - beta_phi_out(radius, time))
# 					vgas[i] -= evol[i]["ifr"][-1] * (
# 						1 - beta_phi_in(radius, time))
# 					vgas[i] *= radius / evol[i]["mgas"][-1]
# 					# else:
# 					# 	vgas[i] = 0
# 					if abs(vgas[i]) > self.dr / self.dt:
# 						sgn_vgas = int(vgas[i] > 0) - int(vgas[i] < 0)
# 						vgas[i] = sgn_vgas * self.dr / self.dt - 1.e9
# 					else: pass
# 				for i in range(n_zones):
# 					radius = self.dr * (i + 0.5)
# 					if radius < MAX_SF_RADIUS and vgas[i] != 0:
# 						dlnmgas_dr = (evol[i + 1]["mgas"][-1] - 
# 							evol[i]["mgas"][-1]) / (
# 							evol[i]["mgas"][-1] * self.dr)
# 						if radius + self.dr < MAX_SF_RADIUS:
# 							dlnvgas_dr = (vgas[i + 1] - vgas[i]) / (
# 								vgas[i] * self.dr)
# 						else:
# 							dlnvgas_dr = 0
# 						mu[i] = -evol[i]["tau_star"][-1] * vgas[i] * (
# 							dlnmgas_dr + dlnvgas_dr)
# 					else:
# 						mu[i] = 0
# 					if mu[i] > 10:
# 						mu[i] = 10
# 					elif mu[i] < -10:
# 						mu[i] = -10
# 					else: pass
# 			else: pass

# 			for i in range(n_zones):
# 				if callable(self.mw_model.zones[i].eta):
# 					eta = self.mw_model.zones[i].eta(time)
# 				else:
# 					eta = self.mw_model.zones[i].eta
# 				mdot = evol[i]["ifr"][-1] - evol[i]["sfr"][-1] * (1 + eta -
# 					mu[i] - recycling)
# 				evol[i]["mgas"].append(evol[i]["mgas"][-1] + mdot * self.dt)
# 				if evol[i]["mgas"][-1] < 1e-12: evol[i]["mgas"][-1] = 1e-12
# 				evol[i]["ifr"].append(self.mw_model.zones[i].func(time) * 1.0e9)
# 				if callable(self.mw_model.zones[i].tau_star):
# 					tau_star = self.mw_model.zones[i].tau_star(time,
# 						evol[i]["mgas"][-1])
# 				else:
# 					tau_star = self.mw_model.zones[i].tau_star
# 				evol[i]["sfr"].append(evol[i]["mgas"][-1] / tau_star)
# 				evol[i]["tau_star"].append(tau_star)
# 				evol[i]["mstar"].append(evol[i]["mstar"][-1] +
# 					evol[i]["sfr"][-1] * (1 - recycling) * self.dt)
# 				evol[i]["ofr"].append(evol[i]["sfr"][-1] * eta)
# 				evol[i]["vgas"].append(vgas[i])
# 				evol[i]["time"].append(time + self.dt)
# 			time += self.dt
# 			sys.stdout.write("\rt = %.2f Gyr" % (time))
# 		sys.stdout.write("\n")
# 		return evol


# 	def normalize(self, recycling = 0.4, tolerance = 0.005):
# 		print("Normalizing....")
# 		while True:
# 			evol = self.evolve(recycling = recycling)
# 			mstar = sum([evol[i]["mstar"][-1] for i in range(len(evol))])
# 			print("MW: %.3e" % (M_STAR_MW))
# 			print("Model: %.3e" % (mstar))
# 			ratio = M_STAR_MW / mstar
# 			success = abs(ratio - 1) < tolerance
# 			if success:
# 				break
# 			else:
# 				ratio = M_STAR_MW / mstar
# 				for i in range(len(self.mw_model._evolution._evol)):
# 					self.mw_model._evolution._evol[i].norm *= ratio
# 				print(ratio)
# 		self._velocities = self.mw_model.n_zones * [None]
# 		for i in range(self.mw_model.n_zones):
# 			self._velocities[i] = interp_scheme_1d(
# 				evol[i]["time"], evol[i]["vgas"])

# 	# def normalize(self, recycling = 0.4, tolerance = 0.05):
# 	# 	n_relevant_zones = int(MAX_SF_RADIUS / self.dr)
# 	# 	print("Normalizing....")
# 	# 	expectations = []
# 	# 	for i in range(self.mw_model.n_zones):
# 	# 		expectations.append(
# 	# 			gradient(self.dr * (i + 0.5)) * np.pi *
# 	# 			self.dr**2 * ((i + 1)**2 - i**2)
# 	# 		)
# 	# 	tot = sum(expectations)
# 	# 	for i in range(self.mw_model.n_zones):
# 	# 		expectations[i] *= M_STAR_MW / tot
# 	# 	while True:
# 	# 		evol = self.evolve(recycling = recycling)
# 	# 		ratios = []
# 	# 		success = True
# 	# 		for i in range(self.mw_model.n_zones):
# 	# 			ratios.append(evol[i]["mstar"][-1] / expectations[i])
# 	# 			if i < n_relevant_zones:
# 	# 				success &= abs(ratios[-1] - 1) < tolerance
# 	# 			else: pass
# 	# 		if success:
# 	# 			break
# 	# 		else:
# 	# 			print("===================")
# 	# 			for i in range(len(self.mw_model._evolution._evol)):
# 	# 				self.mw_model._evolution._evol[i].norm /= ratios[i]
# 	# 				print("R = %.2f kpc ; factor = %.3e" % (
# 	# 					self.dr * (i + 0.5), ratios[i]))
# 	# 	self._velocities = self.mw_model.n_zones * [None]
# 	# 	for i in range(self.mw_model.n_zones):
# 	# 		self._velocities[i] = interp_scheme_1d(
# 	# 			evol[i]["time"], evol[i]["vgas"])




class amd_sfrmode(amd):


	# def __init__(self, mw_model, beta_phi_in = 0.7, beta_phi_out = 0, onset = 1,
	# 	dr = 0.1, dt = 0.01, outfilename = "gasvelocities.out"):
	# 	super().__init__(onset = onset, dr = dr, dt = dt,
	# 		outfilename = outfilename)
	# 	self.mw_model = mw_model
	# 	self.beta_phi_in = beta_phi_in
	# 	self.beta_phi_out = beta_phi_out


	def __call__(self, time):
		radii = [self.dr * i for i in range(int(MAX_RADIUS / self.dr))]
		vgas = len(radii) * [0.]
		crf = vice.cumulative_return_fraction(time)
		for i in range(1, len(radii)):
			if radii[i] <= MAX_SF_RADIUS:
				vgas[i] = vgas[i - 1] + self.dr * self.dvdr(time, radii[i - 1],
					vgas[i - 1], recycling = crf)
			else:
				vgas[i] = 0
		self.write(time, radii, vgas)
		return [radii, vgas]


	def dvdr(self, time, radius, vgas, recycling = 0.4):
		zone = get_bin_number(self.mw_model.annuli, radius)
		if zone < 0: raise ValueError(
			"Radius outside of allowed range: %g" % (radius))

		neighbor = self.mw_model.zones[zone + 1]
		zone = self.mw_model.zones[zone]

		sfr = zone.func(time) * 1.e9 # yr^-1 -> Gyr^-1
		n_sfr = neighbor.func(time) * 1.e9
		sfr_next = zone.func(time + self.dt) * 1.e9

		taustar = zone.tau_star(time, sfr / 1.e9)
		n_taustar = neighbor.tau_star(time, n_sfr / 1.e9)
		taustar_next = zone.tau_star(time + self.dt, sfr_next / 1.e9)

		mg = sfr * taustar
		n_mg = n_sfr * n_taustar
		mg_next = sfr_next * taustar_next
		dlnmgdt = (mg_next - mg) / (mg * self.dt)

		sigmag = mg / (np.pi * ((radius + self.dr)**2 - radius**2))
		n_sigmag = n_mg / (np.pi * ((radius + 2 * self.dr)**2 -
			(radius + self.dr)**2))
		dlnsigmagdr = (n_sigmag - sigmag) / (sigmag * self.dr)

		if callable(zone.eta):
			eta = zone.eta(time)
		else:
			eta = zone.eta
		if callable(self.beta_phi_in):
			beta_phi_in = self.beta_phi_in(radius, time)
		else:
			beta_phi_in = self.beta_phi_in
		if callable(self.beta_phi_out):
			beta_phi_out = self.beta_phi_out(radius, time)
		else:
			beta_phi_out = self.beta_phi_out

		if radius:
			one_over_r = 1 / radius
		else:
			dlnmgdr = (n_mg - mg) / (mg * self.dr)
			one_over_r = dlnmgdr - dlnsigmagdr

		dvdr = 0
		dvdr -= dlnmgdt
		dvdr -= (1 - recycling) / taustar
		dvdr += eta / taustar * (beta_phi_out - beta_phi_in) / (beta_phi_in - 1)
		dvdr -= vgas * (one_over_r * (beta_phi_in - 2) / (beta_phi_in - 1)
			+ dlnsigmagdr)

		return dvdr


class ora(base):

	def __init__(self, mw_model, onset = 1, dr = 0.1, dt = 0.01,
		outfilename = "gasvelocities.out"):
		super().__init__(onset = onset, dr = dr, dt = dt,
			outfilename = outfilename)
		if isinstance(mw_model, vice.milkyway):
			self.mw_model = mw_model
		else:
			raise TypeError(r"""\
Attribute 'mw_model' must be of type vice.milkyway. Got: %s.""" % (
				type(mw_model)))


	def __call__(self, time, recycling = 0.4):
		radii = [self.dr * i for i in range(int(MAX_RADIUS / self.dr))]
		vgas = len(radii) * [0.]
		for i in range(1, len(radii)):
			if radii[i] <= MAX_SF_RADIUS:
				crf = vice.cumulative_return_fraction(time)
				vgas[i] = self.next_vgas(vgas[i - 1], time, radii[i - 1],
					recycling = crf)
			else:
				vgas[i] = 0
		self.write(time, radii, vgas)
		return [radii, vgas]


	def next_vgas(self, vgas, time, radius, recycling = 0.4):
		zone = get_bin_number(self.mw_model.annuli, radius)
		if zone < 0: raise ValueError(
			"Radius outside of allowed range: %g" % (radius))

		radius = self.mw_model.annuli[zone] # use inner edge
		if zone == len(self.mw_model.annuli) - 1: return 0
		neighbor = self.mw_model.zones[zone + 1]
		zone = self.mw_model.zones[zone]

		sfr = zone.func(time) * 1.e9 # yr^-1 -> Gyr^-1
		n_sfr = neighbor.func(time) * 1.e9

		taustar = zone.tau_star(time, sfr / 1.e9)
		n_taustar = neighbor.tau_star(time, n_sfr / 1.e9)

		mgas = sfr * taustar
		n_mgas = n_sfr * n_taustar

		sfr_next = zone.func(time + self.dt) * 1.e9
		mgas_next = sfr_next * zone.tau_star(time + self.dt, sfr_next / 1.e9)

		if callable(zone.eta):
			eta = zone.eta(time)
		else:
			eta = zone.eta

		x = vgas**2 * self.dt**2 - 2 * radius * vgas * self.dt
		x /= 2 * radius * self.dr + self.dr**2
		x *= mgas
		x += mgas_next - mgas
		x += sfr * self.dt * (1 + eta - recycling) 
		x *= 2 * radius * self.dr + 3 * self.dr**2
		x /= n_mgas
		x += (radius + self.dr)**2

		if x < 0: raise ValueError("x < 0: %.5e. r = %.5e. t = %.5e" % (x,
			radius, time))

		n_vgas = 1 / self.dt * (radius + self.dr - np.sqrt(x))
		return n_vgas

