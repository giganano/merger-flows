r"""
This file implements the normalization calculation in Appendix A of
Johnson et al. (2021).
"""

from ..._globals import MAX_SF_RADIUS, END_TIME, M_STAR_MW
import numpy as np
import math as m


def normalize_ifrmode(mwmodel, gradient, dr = 0.1, dt = 0.01, tolerance = 0.1):
	n_relevant_zones = int(MAX_SF_RADIUS / dr)
	expectations = []
	for i in range(mwmodel.n_zones):
		expectations.append(
			gradient(dr * (i + 0.5)) * dr**2 * ((i + 1)**2 - i**2))
	for i in range(mwmodel.n_zones):
		expectations[i] *= M_STAR_MW / sum(expectations)
	dt_for_norm = 0.05
	elements = mwmodel.elements
	mwmodel.dt = dt_for_norm
	# mwmodel.elements = []
	while True:
		out = mwmodel.run(
			np.linspace(0, END_TIME, int(END_TIME / dt_for_norm)),
			overwrite = True,
			capture = True)
		mwmodel.radialflow.last_output = out
		success = True
		for i in range(n_relevant_zones):
			ratio = out.zones["zone%d" % (i)].history["mstar"][-1]
			ratio /= expectations[i]
			success &= abs(ratio - 1) < tolerance
		if success:
			break
		else:
			for i in range(n_relevant_zones):
				ratio = expectations[i]
				ratio /= out.zones["zone%d" % (i)].history["mstar"][-1]
				# print(mwmodel.zones[i].func.surface_density)
				mwmodel.zones[i].func.surface_density._evol[i].norm *= ratio
	mwmodel.dt = dt
	# mwmodel.elements = elements







# def normalize_ifrmode(mwmodel, gradient, dr = 0.1, dt = 0.01):
# 	# timestep_size = mwmodel.zones[0].dt
# 	dt_for_norm = 0.01 # timestep size for sake of normalization
# 	mwmodel.dt = dt_for_norm
# 	print("Normalizing....")
# 	ism_evol = mwmodel.run(
# 		np.linspace(0, END_TIME, int(END_TIME / dt_for_norm)),
# 		ism_only = True)
# 	print("Finished ISM-only dry run!")
# 	expectations = []
# 	for i in range(mwmodel.n_zones):
# 		expectations.append(gradient(dr * (i + 0.5)) * 
# 			((dr * (i + 1))**2 - (dr * i)**2))
# 	for i in range(mwmodel.n_zones):
# 		expectation[i] *= M_STAR_MW / sum(expectations)
# 	prefactors = []
# 	for i in range(mwmodel.n_zones):
# 		mstar = ism_evol["zone%d" % (i)]["mstar"][-1]
# 		prefactors.append(expectations[i] / mstar)
# 	mwmodel.dt = dt
# 	return prefactors

	# prefactors = []
	# mstar = 0
	# for i in range(mwmodel.n_zones):
	# 	if i == mwmodel.n_zones - 1: break
	# 	mass = ism_evol["zone%d" % (i)]["mstar"][-1]
	# 	mstar += mass
	# 	area = m.pi * ((dr * (i + 1))**2 - (dr * i)**2)
	# 	sigma = mass / area
	# 	mass_next = ism_evol["zone%d" % (i + 1)]["mstar"][-1]
	# 	area_next = m.pi * ((dr * (i + 2))**2 - (dr * (i + 1))**2)
	# 	sigma_next = mass_next / area_next
	# 	dsigma_dr = (sigma_next - sigma) / dr
	# 	expected = (gradient)


def normalize(time_dependence, radial_gradient, radius, dt = 0.01, dr = 0.5,
	recycling = 0.4):
	r"""
	Determine the prefactor on the surface density of star formation as a
	function of time as described in Appendix A of Johnson et al. (2021).

	Parameters
	----------
	time_dependence : <function>
		A function accepting time in Gyr and galactocentric radius in kpc, in
		that order, specifying the time-dependence of the star formation
		history at that radius. Return value assumed to be unitless and
		unnormalized.
	radial_gradient : <function>
		A function accepting galactocentric radius in kpc specifying the
		desired stellar radial surface density gradient at the present day.
		Return value assumed to be unitless and unnormalized.
	radius : real number
		The galactocentric radius to evaluate the normalization at.
	dt : real number [default : 0.01]
		The timestep size in Gyr.
	dr : real number [default : 0.5]
		The width of each annulus in kpc.
	recycling : real number [default : 0.4]
		The instantaneous recycling mass fraction for a single stellar
		population. Default is calculated for the Kroupa IMF [1]_.

	Returns
	-------
	A : real number
		The prefactor on the surface density of star formation at that radius
		such that when used in simulation, the correct total stellar mass with
		the specified radial gradient is produced.

	Notes
	-----
	This function automatically adopts the desired maximum radius of star
	formation, end time of the model, and total stellar mass declared in
	``src/_globals.py``.

	.. [1] Kroupa (2001), MNRAS, 322, 231
	"""

	time_integral = 0
	for i in range(int(END_TIME / dt)):
		time_integral += time_dependence(i * dt) * dt * 1.e9 # yr to Gyr

	radial_integral = 0
	for i in range(int(MAX_SF_RADIUS / dr)):
		radial_integral += radial_gradient(dr * (i + 0.5)) * m.pi * (
			(dr * (i + 1))**2 - (dr * i)**2
		)

	return M_STAR_MW / ((1 - recycling) * radial_integral * time_integral)

