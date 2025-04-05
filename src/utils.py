
import numpy as np
import vice
import os


def oh_to_12pluslog(oh, solaro = vice.solar_z["o"], mo = 15.999, Xsun = 0.73):
	# return 12 + np.log10(mh / mo * solaro * 10**oh)
	return 12 + np.log10(solaro / (Xsun * mo)) + oh


def get_velocity_profile(output, lookback):
	raw = np.genfromtxt("%s_gasvelocities.out" % (output.name))
	time = output.zones["zone0"].history["time"][-1] - lookback
	diff = [abs(_ - time) for _ in output.zones["zone0"].history["time"]]
	idx = diff.index(min(diff))
	time = output.zones["zone0"].history["time"][idx]
	radii = []
	vgas = []
	for i in range(len(raw)):
		if raw[i][0] == time:
			radii.append(raw[i][1])
			vgas.append(raw[i][2])
		else: pass
	return [radii, vgas]


def get_velocity_evolution(output, radius, zone_width = 0.1):
	if os.path.exists("%s_gasvelocities.out" % (output.name)):
		raw = np.genfromtxt("%s_gasvelocities.out" % (output.name))
	else:
		lookback = output.zones["zone0"].history["lookback"]
		vgas = len(lookback) * [0.]
		return [lookback, vgas]
	zone = int(radius / zone_width)
	radius = zone * zone_width # use inner edge for sake of lookup in file
	time = []
	vgas = []
	for i in range(len(raw)):
		if raw[i][1] == radius:
			time.append(raw[i][0])
			vgas.append(raw[i][2])
		else: pass
	lookback = [time[-1] - t for t in time]
	return [lookback, vgas]


	# raw = np.genfromtxt("%s_gasvelocities.out" % (output.name))
	# raw = vice.dataframe({
	# 	"time": 	raw[:, 0],
	# 	"radius": 	raw[:, 1],
	# 	"vgas": 	raw[:, 2]
	# 	})
	# diff = [abs(_ - lookback) for _ in output.zones["zone0"].history["lookback"]]
	# idx = diff.index(min(diff))
	# time = output.zones["zone0"].history["time"][idx]
	# raw = raw.filter("time", "==", time)
	# return [raw["radius"], raw["vgas"]]
	# raw = np.genfromtxt("%s_gasvelocities.out" % (output.name))
	# diff = [abs(_ - lookback) for _ in output.zones["zone0"].history["lookback"]]
	# idx = diff.index(min(diff))
	# time = output.zones["zone0"].history["time"][idx]
	# radii = []
	# vgas = []
	# if idx < len(output.zones["zone0"].history["time"]) - 1:
	# 	dt = output.zones["zone0"].history["time"][idx + 1] - time
	# else:
	# 	dt = time - output.zones["zone0"].history["time"][idx - 1]

	# indices = np.where(raw[:,0] == time)
	# if len(indices):
	# 	radii = [raw[idx][1] for idx in indices]
	# 	vgas = [raw[idx][2] for idx in indices]
	# else: pass
	# return [radii, vgas]

def mu(output, lookback, zone_width = 0.1):
	radii, vgas = get_velocity_profile(output, lookback)
	diff = [abs(_ - lookback) for _ in output.zones["zone0"].history["lookback"]]
	idx = diff.index(min(diff))
	mu_gas = []
	mu_oxygen = []
	for i in range(len(radii) - 1):
		zone = output.zones["zone%d" % (i)]
		neighbor = output.zones["zone%d" % (i + 1)]
		area = np.pi * ((radii[i] + zone_width)**2 - radii[i]**2)
		n_area = np.pi * ((radii[i + 1] + zone_width)**2 - radii[i + 1]**2)
		if radii[i + 1] >= 15.5:
			mu_gas.append(float("nan"))
			mu_oxygen.append(float("nan"))
		else:
			if zone.history["sfr"][idx]:
				tau_star = zone.history["mgas"][idx] / zone.history["sfr"][idx]
				tau_star *= 1.e-9
			else:
				tau_star = float("inf")
			mu = (neighbor.history["mgas"][idx] - zone.history["mgas"][idx]) / (
				zone.history["mgas"][idx] * zone_width)
			mu += (vgas[i + 1] - vgas[i]) / (vgas[i] * zone_width)
			mu *= -tau_star * vgas[i]
			# mu = -tau_star * vgas[i] / radii[i] if radii[i] else float("inf")
			# sigma_g = zone.history["mgas"][idx] / area
			# n_sigma_g = neighbor.history["mgas"][idx] / area
			# mu -= tau_star * vgas[i] * (n_sigma_g - sigma_g) / (
			# 	sigma_g * zone_width)
			# mu -= tau_star * vgas[i] * (vgas[i + 1] - vgas[i]) / (
			# 	vgas[i] * zone_width)
			mu_gas.append(mu)
			mu -= tau_star * vgas[i] * (
				neighbor.history["z(o)"][idx] - zone.history["z(o)"][idx]) / (
				zone.history["z(o)"][idx] * zone_width)
			mu_oxygen.append(mu)
	return [radii, mu_gas, mu_oxygen]


def mu_evol(output, radius, zone_width = 0.1):
	lookback, vgas = get_velocity_evolution(output, radius,
		zone_width = zone_width)
	if all([v == 0 for v in vgas]):
		mu_gas = mu_o = len(lookback) * [0.]
		return [lookback, mu_gas, mu_o]
	else: pass
	_, vgas_n = get_velocity_evolution(output, radius + zone_width,
		zone_width = zone_width)
	zone = int(radius / zone_width)
	neighbor = output.zones["zone%d" % (zone + 1)]
	zone = output.zones["zone%d" % (zone)]
	mu_gas = []
	mu_o = []
	for i in range(len(lookback)):
		# idx = -1 - i
		diff = [abs(lookback[i] - l) for l in zone.history["lookback"]]
		idx = diff.index(min(diff))
		if zone.history["sfr"][idx]:
			tau_star = zone.history["mgas"][idx] / zone.history["sfr"][idx]
			tau_star *= 1.e-9
		else:
			mu_gas.append(float("nan"))
			mu_o.append(float("nan"))
			continue
		mu = (neighbor.history["mgas"][idx] - zone.history["mgas"][idx]) / (
			zone.history["mgas"][idx] * zone_width)
		mu += (vgas_n[i] - vgas[i]) / (vgas[i] * zone_width)
		mu *= -tau_star * vgas[i]
		mu_gas.append(mu)
		mu -= tau_star * vgas[i] * (
			neighbor.history["z(o)"][idx] - zone.history["z(o)"][idx]) / (
			zone.history["z(o)"][idx] * zone_width)
		mu_o.append(mu)
	return [lookback, mu_gas, mu_o]


# def mu_evolution(output, radius, zone_width = 0.1):
# 	zone = int(radius / zone_width)
# 	radius = zone_width * radius # use inner edge of zone
# 	neighbor = output.zones["zone%d" % (zone + 1)]
# 	zone = output.zones["zone%d" % (zone)]
# 	mu_gas = []
# 	mu_oxygen = []
# 	time = zone.history["time"]
# 	for 


def boxcarsmoothtrend(xvals, yvals, window = 10):
	assert len(xvals) == len(yvals), "Array-length mismatch: (%d, %d)" % (
		len(xvals), len(yvals))
	smoothed = len(xvals) * [0.]
	for i in range(len(xvals)):
		start = max(0, i - window)
		stop = min(i + window, len(xvals) - 1)
		smoothed[i] = np.mean(yvals[start:stop])
	return smoothed

