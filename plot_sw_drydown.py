#!/usr/bin/env python

"""
Plot response to idealised dry-down

"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (02.02.2020)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import xarray as xr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('src')

import constants as c
import parameters as p
from radiation import calculate_absorbed_radiation
from two_leaf import Canopy as TwoLeaf

#import warnings
#warnings.filterwarnings("ignore")

def main(p, met, lai):

    days = met.doy
    hod = met.hod
    ndays = int(len(days) / 24.)
    nhours = len(met)
    hours_in_day = int(nhours / float(ndays))

    if hours_in_day == 24:
        met_timestep = 60.
    else:
        met_timestep = 30.
    timestep_sec = 60. * met_timestep


    T = TwoLeaf(p, gs_model="medlyn")
    #T = TwoLeaf(p, gs_model="leuning")

    out, store = setup_output_dataframe(ndays, hours_in_day, p)

    i = 0
    hour_cnt = 1 # hour count
    day_cnt = 0
    while i < len(met):
        year = met.index.year[i]
        doy = met.doy[i]
        hod = met.hod[i] + 1

        if day_cnt-1 == -1:
            beta = calc_beta(p.theta_sat, p.theta_fc, p.theta_wp)
        else:
            beta = calc_beta(out.sw[day_cnt-1], p.theta_fc, p.theta_wp)
            #beta = calc_beta(store.sw[hour_cnt-1])

        (An, et, Tcan,
         apar, lai_leaf) = T.main(met.tair[i], met.par[i], met.vpd[i],
                                  met.wind[i], met.press[i], met.ca[i],
                                  doy, hod, lai[i], beta=beta)

        if hour_cnt == hours_in_day: # End of the day

            store_hourly(hour_cnt-1, An, et, lai_leaf, met.precip[i], store,
                         hours_in_day)

            store_daily(year, doy, day_cnt, store, beta, out, p)

            hour_cnt = 1
            day_cnt += 1
        else:
            store_hourly(hour_cnt-1, An, et, lai_leaf, met.precip[i], store,
                         hours_in_day)

            hour_cnt += 1

        i += 1

    return (out)

def calc_beta(theta, theta_fc, theta_wp):
    q = 1.0#0.5

    beta = ((theta - theta_wp) / (theta_fc - theta_wp))**q
    #print(theta, beta)
    beta = max(0.0, beta)
    beta = min(1.0, beta)

    return beta

def setup_output_dataframe(ndays, nhours, p):

    zero = np.zeros(ndays)
    out = pd.DataFrame({'year':zero, 'doy':zero,
                        'An_can':zero, 'E_can':zero, 'LAI':zero, 'sw':zero,
                        'beta': zero})

    out.sw[0] = p.theta_sat

    zero = np.zeros(nhours)
    hour_store = pd.DataFrame({'An_can':zero, 'E_can':zero, 'LAI_can':zero,
                               'delta_sw':zero, 'sw':zero})
    hour_store.sw[0] = p.theta_sat
    return (out, hour_store)

def store_hourly(idx, An, et, lai_leaf, precip, store, hours_in_day):

    if hours_in_day == 24:
        an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * c.SEC_TO_HR
        et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HR
        precip_conv = c.SEC_TO_HR
        met_timestep = 60.
    else:
        an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * c.SEC_TO_HLFHR
        et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HLFHR
        precip_conv = c.SEC_TO_HLFHR
        met_timestep = 30.

    #sun_frac = lai_leaf[c.SUNLIT] / np.sum(lai_leaf)
    #sha_frac = lai_leaf[c.SHADED] / np.sum(lai_leaf)
    store.An_can[idx] = np.sum(An * an_conv)
    store.E_can[idx] = np.sum(et * et_conv)
    store.LAI_can[idx] = np.sum(lai_leaf)

    precip *= precip_conv
    #precip_max = 1.0    # hack for runoff
    #if precip > precip_max:
    #    precip = precip_max

    store.delta_sw[idx] = precip - store.E_can[idx]

    #timestep_sec = 60. * met_timestep
    #store.delta_sw[idx] = (precip * timestep_sec) - store.E_can[idx]
    #if idx-1 == -1:
    #    prev_sw = p.theta_sat
    #else:
    #    prev_sw = store.sw[idx-1]
    #store.sw[idx] = update_sw_bucket(p, store.delta_sw[idx], prev_sw)

def store_daily(year, doy, idx, store, beta, out, p):

    out.year[idx] = year
    out.doy[idx] = doy
    out.An_can[idx] = np.sum(store.An_can)
    out.E_can[idx] = np.sum(store.E_can)
    out.LAI[idx] = np.mean(store.LAI_can)
    out.beta[idx] = beta
    #out.sw[idx] = store.sw.iloc[-1]

    #"""
    if idx-1 == -1:
        prev_sw = p.theta_sat
    else:
        prev_sw = out.sw[idx-1]


    out.sw[idx] = update_sw_bucket(p, np.sum(store.delta_sw), prev_sw)
    #"""

def update_sw_bucket(p, delta_sw, sw_prev):
    """
    Update the simple bucket soil water balance

    Parameters:
    -----------
    precip : float
        precipitation (kg m-2 s-1)
    water_loss : float
        flux of water out of the soil (transpiration (kg m-2 timestep-1))
    sw_prev : float
        volumetric soil water from the previous timestep (m3 m-3)

    Returns:
    -------
    sw : float
        new volumetric soil water (m3 m-3)
    """

    sw = min(p.theta_sat, \
             sw_prev + delta_sw / (p.soil_volume * c.M_2_MM))
    #print(sw, sw_prev, delta_sw / (p.soil_volume * c.M_2_MM))
    #sw = max(0.0, sw)
    #print(sw)
    return sw



def read_met_file(fname):
    """ Build a dataframe from the netcdf outputs """

    ds = xr.open_dataset(fname)
    lat = ds.latitude.values[0][0]
    lon = ds.longitude.values[0][0]

    vars_to_keep = ['SWdown','Tair','Wind','Psurf',\
                    'VPD','CO2air','Precip']
    df = ds[vars_to_keep].squeeze(dim=["y","x"],
                                  drop=True).to_dataframe()

    time_idx = df.index

    df = df.reindex(time_idx)
    df['year'] = df.index.year
    df['hod'] = df.index.hour
    df['doy'] = df.index.dayofyear
    df["par"] = df.SWdown * c.SW_2_PAR
    df["Tair"] -= c.DEG_2_KELVIN
    df = df.drop('SWdown', axis=1)
    df = df.rename(columns={'PAR': 'par', 'Tair': 'tair', 'Wind': 'wind',
                            'VPD': 'vpd', 'CO2air': 'ca', 'Psurf': 'press',
                            'Precip': 'precip'})

    # Make sure there is no bad data...
    df.vpd = np.where(df.vpd < 0.0, 0.0, df.vpd)

    return df, lat, lon


if __name__ == "__main__":


    #
    ##  Just use TUMBA nc for now
    #
    met_fn = "met/AU-Tum_2002-2017_OzFlux_Met.nc"
    (met, lat, lon) = read_met_file(met_fn)

    #plt.plot(met.press)
    #plt.show()

    # Just keep ~ a spring/summer
    #met = met[(met.index.year == 2003) | (met.index.year == 2004)]
    #met = met[ ((met.index.year == 2003) & (met.doy >= 260)) |
    #           ((met.index.year == 2004) & (met.doy <= 90)) ]


    #"""
    # Just keep a summer
    met = met[ ((met.index.year == 2003) & (met.doy >= 120)) &
               ((met.index.year == 2003) & (met.doy <= 260)) ]

    # no precip
    #met.precip[ ((met.index.year == 2003) & (met.doy >= 330)) |
    #            ((met.index.year == 2003) & (met.doy <= 365)) ] /= 2.0

    #met.precip[((met.index.year == 2003) & (met.doy >= 342)) ] = 0.0
    #met.precip[((met.index.year == 2003) & (met.doy <= 338)) ] = 0.0

    met.precip = 0.0

    # Fix met data
    aco2 = 400
    eco2 = aco2 * 2.
    alai = 2.0
    elai = alai * 2.
    avpd = 3.0
    evpd = avpd * 1.5

    met.ca = aco2
    #met.tair = 25.
    #met.vpd = 3.
    #met.par = 1500.
    #met.wind = 3.
    met.press = 101325.0 # Pa

    # Day
    met.tair[((met.index.hour >= 5) & (met.index.hour <= 19)) ] = 25.
    met.vpd[((met.index.hour >= 5) & (met.index.hour <= 19)) ] = avpd
    met.par[((met.index.hour >= 5) & (met.index.hour <= 19)) ] = 2000.
    met.wind[((met.index.hour >= 5) & (met.index.hour <= 19)) ] = 3.

    # Night
    met.tair[((met.index.hour < 5) & (met.index.hour > 19)) ] = 25.
    met.vpd[((met.index.hour < 5) & (met.index.hour > 19)) ] = 0.
    met.par[((met.index.hour < 5) & (met.index.hour > 19)) ] = 0.
    met.wind[((met.index.hour < 5) & (met.index.hour > 19)) ] = 0.

    #"""

    #for i in met.precip:
    #    print(i)
    #sys.exit()
    time = met.copy()
    time_day = time.resample('D').mean()
    time_day = time_day.index


    #aCa = met.ca.values.copy()
    #eCa = aCa * 1.6
    #print(np.mean(aCa), np.mean(eCa))

    #print( np.sum(met.precip * 3600.0) )
    #sys.exit()


    lai = np.ones(len(met)) * alai
    out_aCa = main(p, met, lai)

    lai = np.ones(len(met)) * elai
    out_aCa_eL = main(p, met, lai)

    met.ca = eco2
    lai = np.ones(len(met)) * alai
    out_eCa = main(p, met, lai)

    lai = np.ones(len(met)) * elai
    out_eCa_eL = main(p, met, lai)

    met.vpd = evpd
    out_eCa_eL_eD = main(p, met, lai)

    met.vpd = evpd
    lai = np.ones(len(met)) * alai
    out_eCa_aL_eD = main(p, met, lai)

    met.vpd = evpd
    met.ca = eco2
    lai = np.ones(len(met)) * alai
    out_eCa_aL_eD = main(p, met, lai)

    win=3


    fig = plt.figure(figsize=(9,12))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.1)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)


    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa.E_can.rolling(window=win).mean(),
             c="black", lw=3.0, ls="-", label="aCO$_2$")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa.E_can.rolling(window=win).mean(),
             c=colours[2], lw=3.0, ls="-", label="eCO$_2$")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa_eL.E_can.rolling(window=win).mean(),
             c=colours[1], lw=3.0, ls="-", label="eLAI")
    #ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         out_aCa_aL_eD.E_can.rolling(window=win).mean(),
    #         c=colours[5], lw=3.0, ls="-", label="eVPD")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_eL.E_can.rolling(window=win).mean(),
             c=colours[0], lw=3.0, ls="-", label="eCO$_2$ & eLAI")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_eL_eD.E_can.rolling(window=win).mean(),
             c=colours[4], lw=3.0, ls="-", label="eCO$_2$ & eLAI & eVPD")

    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa.beta.rolling(window=win).mean(),
             c="black", lw=3.0, ls="-", label="aCO$_2$")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa.beta.rolling(window=win).mean(),
             c=colours[2], lw=3.0, ls="-", label="eCO$_2$")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa_eL.beta.rolling(window=win).mean(),
             c=colours[1], lw=3.0, ls="-", label="eLAI")
    #ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
    ##         out_aCa_aL_eD.beta.rolling(window=win).mean(),
    #         c=colours[5], lw=3.0, ls="-", label="eVPD")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_eL.beta.rolling(window=win).mean(),
             c=colours[0], lw=3.0, ls="-", label="eCO$_2$ & eLAI")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_eL_eD.beta.rolling(window=win).mean(),
             c=colours[4], lw=3.0, ls="-", label="eCO$_2$ & eLAI & eVPD")

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel("Transpiration (mm d$^{-1}$)", fontsize=14)
    ax2.set_ylabel(r"$\beta$ (-)")
    ax2.set_xlabel("Days in drought", fontsize=14)
    #ax1.legend(numpoints=1, loc="best", frameon=False)
    ax2.legend(numpoints=1, loc="best", frameon=False)

    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    #fig.autofmt_xdate()
    #fig.savefig("drydown.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    fig.savefig("raw_E.pdf", bbox_inches='tight', pad_inches=0.1)






    fig = plt.figure(figsize=(9,12))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.1)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_eCa.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[2], lw=3.0, ls="-", label="eCO$_2$")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_aCa_eL.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[1], lw=3.0, ls="-", label="eLAI")
    #ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         (out_aCa_aL_eD.E_can / out_aCa.E_can).rolling(window=win).mean(),
    #         c=colours[5], lw=3.0, ls="-", label="eVPD")

    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_eCa_eL.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[0], lw=3.0, ls="-", label="eCO$_2$ & LAI")
    #ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         (out_eCa_eL_eD.E_can / out_aCa.E_can).rolling(window=win).mean(),
    #         c=colours[4], lw=3.0, ls="-", label="eCO$_2$ & eLAI & eVPD")
    ax1.axhline(y=1.0, ls="--", color="lightgrey", alpha=0.5)

    ax1.set_ylim(0.2, 1.5)
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    #plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa.sw.rolling(window=win).mean(), c="black", lw=3.0,
             ls="-", label="aCO$_2$")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa.sw.rolling(window=win).mean(), c=colours[2], lw=3.0,
             ls="-", label="eCO$_2$")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa_eL.sw.rolling(window=win).mean(), c=colours[1], lw=3.0,
             ls="-", label="eLAI")
    #ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         out_aCa_aL_eD.sw.rolling(window=win).mean(), c=colours[5], lw=3.0,
    #         ls="-", label="eVPD")

    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_eL.sw.rolling(window=win).mean(), c=colours[0],
             lw=3.0, ls="-", label="eCO$_2$ & eLAI")
    #ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         out_eCa_eL_eD.sw.rolling(window=win).mean(), c=colours[4],
    #         lw=3.0, ls="-", label="eCO$_2$ & eLAI & eVPD")
    #ax2.set_xlim(0, 18)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel("Transpiration (e$C_{a}$/a$C_{a}$)", fontsize=14)
    ax2.set_ylabel(r"$\theta$ (m$^{3}$ m$^{-3}$)")
    ax2.set_xlabel("Days in drought", fontsize=14)

    #ax1.legend(numpoints=1, loc="best", frameon=False)
    ax2.legend(numpoints=1, loc="best", frameon=False)

    from matplotlib.ticker import FormatStrFormatter
    #ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    props = dict(boxstyle='round', facecolor='white', alpha=1.0,
                 ec="white")
    fig_label = "%s" % ("(a)")
    ax1.text(0.015, 0.07, fig_label,
            transform=ax1.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)
    fig_label = "%s" % ("(b)")
    ax2.text(0.015, 0.07, fig_label,
            transform=ax2.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)

    #fig.autofmt_xdate()
    #fig.savefig("drydown.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    fig.savefig("drydown.pdf", bbox_inches='tight', pad_inches=0.1)


    fig = plt.figure(figsize=(9,12))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.1)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_eCa.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[2], lw=3.0, ls="-", label="eCO$_2$")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_aCa_eL.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[1], lw=3.0, ls="-", label="eLAI")
    #ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         (out_aCa_aL_eD.E_can / out_aCa.E_can).rolling(window=win).mean(),
    #         c=colours[5], lw=3.0, ls="-", label="eVPD")

    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_eCa_eL.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[0], lw=3.0, ls="-", label="eCO$_2$ & LAI")
    ax1.plot(np.arange(-1, len(out_eCa.E_can)-1),
             (out_eCa_aL_eD.E_can / out_aCa.E_can).rolling(window=win).mean(),
             c=colours[4], lw=3.0, ls="-", label="eCO$_2$ & eVPD")
    ax1.axhline(y=1.0, ls="--", color="lightgrey", alpha=0.5)

    ax1.set_ylim(0.2, 1.5)
    ax1.set_xlim(0, 50)
    ax2.set_xlim(0, 50)
    #plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa.sw.rolling(window=win).mean(), c="black", lw=3.0,
             ls="-", label="Ambient")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa.sw.rolling(window=win).mean(), c=colours[2], lw=3.0,
             ls="-", label="eCO$_2$")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_aCa_eL.sw.rolling(window=win).mean(), c=colours[1], lw=3.0,
             ls="-", label="eLAI")
    #ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
    #         out_aCa_aL_eD.sw.rolling(window=win).mean(), c=colours[5], lw=3.0,
    #         ls="-", label="eVPD")

    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_eL.sw.rolling(window=win).mean(), c=colours[0],
             lw=3.0, ls="-", label="eCO$_2$ & eLAI")
    ax2.plot(np.arange(-1, len(out_eCa.E_can)-1),
             out_eCa_aL_eD.sw.rolling(window=win).mean(), c=colours[4],
             lw=3.0, ls="-", label="eCO$_2$ & eVPD")
    #ax2.set_xlim(0, 18)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel("Transpiration (elevated/ambient)", fontsize=14)
    ax2.set_ylabel(r"$\theta$ (m$^{3}$ m$^{-3}$)")
    ax2.set_xlabel("Days in drought", fontsize=14)

    #ax1.legend(numpoints=1, loc="best", frameon=False)
    ax2.legend(numpoints=1, loc="best", frameon=False)

    from matplotlib.ticker import FormatStrFormatter
    #ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    props = dict(boxstyle='round', facecolor='white', alpha=1.0,
                 ec="white")
    fig_label = "%s" % ("(a)")
    ax1.text(0.015, 0.07, fig_label,
            transform=ax1.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)
    fig_label = "%s" % ("(b)")
    ax2.text(0.015, 0.07, fig_label,
            transform=ax2.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)

    #fig.autofmt_xdate()
    fig.savefig("drydown_with_D.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    #fig.savefig("drydown_with_D.pdf", bbox_inches='tight', pad_inches=0.1)
