# A single spot model

import numpy as np
import matplotlib.pyplot as plt


def spot(pars, time):
    """
    Period = rotation period.
    inc = inclination angle.
    lon = spot longitude.
    lat = spot latitude.
    """

    period, inc, size, lon, lat, = pars
    flux = np.ones(len(time))

    phase0 = 2*np.pi*time/period
    phase = 2*np.pi*time/period + lon
    # mu = np.cos(inc) * np.sin(lat) + np.sin(inc) * np.cos(lat) \
        # * np.cos(phase)
    mu = np.pi**2 * np.cos(phase)*np.cos(lat + inc)

    phase_mod = phase0 % (2*np.pi)
    m1 = (phase_mod > np.pi*3./2)
    m2 = (phase_mod < np.pi/2)
    # m2 = (phase_mod[m1] < np.pi*2)
    flux -= size*mu
    flux[m1] = np.ones(len(flux[m1]))
    flux[m2] = np.ones(len(flux[m2]))

    plt.clf()
    plt.plot(time, flux, "k.")
    # plt.plot(phase_mod/np.pi, flux, "k.")
    plt.savefig("test")

    # area = amax * np.exp(-(time - pk)**2 / 2. / decay**2)


def two_spot(pars, time):
    """
    Period = rotation period.
    inc = inclination angle.
    lon = spot longitude.
    lat = spot latitude.
    """

    period, inc, size1, size2, lon1, lon2, lat1, lat2 = pars
    flux = np.ones(len(time))

    phase1 = 2*np.pi*time/period + lon1
    phase2 = 2*np.pi*time/period + lon2
    mu1 = np.cos(inc) * np.sin(lat1) + np.sin(inc) * np.cos(lat1) \
        * np.cos(phase1)
    mu2 = np.cos(inc) * np.sin(lat2) + np.sin(inc) * np.cos(lat2) \
        * np.cos(phase2)
    flux -= (size1 * mu1 + size2 * mu2)

    plt.clf()
    plt.plot(time, flux, "k.")
    plt.savefig("test")

    # area = amax * np.exp(-(time - pk)**2 / 2. / decay**2)

if __name__ == "__main__":
    period = 10
    inc = 0
    size = .2
    size2 = .1
    lon = np.pi
    lon2 = 0
    lat = np.pi/3
    lat2 = np.pi/6
    spot([period, inc, size, lon, lat], np.arange(1, 100, .1))
    # two_spot([period, inc, size, size2, lon, lon2, lat, lat2],
    #          np.arange(1, 100, .1))
