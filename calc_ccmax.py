from obspy import read, UTCDateTime
import os
import sys
import shutil
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from obspy.signal.trigger import recursive_sta_lta, trigger_onset, plot_trigger
from obspy.signal.cross_correlation import xcorr_pick_correction
from scipy import signal
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel
from function_cc_all import cc_cent_custom, find_eq, plot_cc_map, plot_cc_map_zoom
import warnings

warnings.filterwarnings("ignore")

# Disable SAC copyright display
os.putenv("SAC_DISPLAY_COPYRIGHT", '0')

# Input event/link files
fevent = 'link_all.txt'
with open(fevent, 'r') as f:
    lines = f.readlines()

fevent_all = 'large_all_event.txt'
with open(fevent_all, 'r') as f_all:
    lines_all = f_all.readlines()

# Build a list of event IDs from the large event file (index 10 in each line)
evid_all = []
for line in lines_all:
    evid0 = int(line.split(' ')[10])
    evid_all.append(evid0)

# Parameters and paths
ixx = 1
num_j = 0  # Maximum number of events to process (set to 0 to process only the first event)
t_window = 0.2
slope_num = 70
seg_t_start = 60
seg_t_end = 180

str_file = 'result/cc_file/'
str_png = 'result/cc_0.8/'
str_png1 = 'result/cc_0.9/'
str_png_zoom = 'result/cc_0.8_zoom/'
str_png_zoom1 = 'result/cc_0.9_zoom/'

str_front_la = '../Events_la/'
str_front_sm = '../Events_sm/'

str_cc_name0 = str_file + 'ccmax_' + str(num_j) + '.txt'
f0 = open(str_cc_name0, 'w')
str_cc_st = str_file + 'st_' + str(num_j) + '.txt'
fst = open(str_cc_st, 'w')

# Time shift arrays for cross-correlation (in seconds)
delta_tj = np.arange(-0.2, 0.1, 0.01)
delta_tjj = np.arange(-0.2, 0.1, 0.01)
delta_tc = np.arange(-0.1, 0.1, 0.01)
snr_vl = 0.1
len_tj_tjj = len(delta_tjj) * len(delta_tj)

# Initialize event counter
ilo = 0

# Loop over all lines in the link file
for ii in range(len(lines)):
    if ilo <= num_j:
        line = lines[ilo].strip()
        ilo += 1

        # Parse event IDs and station information from the line
        parts = line.replace(",", "").replace("'", "").replace("[", "").replace("]", "").replace("  ", " ").split()
        evid_la = int(parts[0])
        evid_sm = int(parts[1])

        # Determine the large event pzdata path; if not found, try the small event directory
        path_la = str_front_la + str(evid_la) + '/pzdata/'
        if not os.path.exists(path_la):
            path_la = str_front_sm + str(evid_la) + '/pzdata/'
        path_sm = str_front_sm + str(evid_sm) + '/pzdata/'

        # The next 10 tokens are network names; the following 10 are station names; then 10 channel names.
        st_network = parts[2:12]
        st_station = parts[12:22]
        st_channel = parts[22:32]

        li = list(range(len(st_channel)))
        ccmax = []
        ccmax_inf = []

        # Loop over delta_tc values (no clip-related check is performed here)
        for ltc in delta_tc:
            cc_inf = []
            cc_max_station_all = []
            for jj in li:
                network = st_network[jj]
                station = st_station[jj]
                channel = st_channel[jj]
                path_la_st = os.path.join(path_la, f"{network}.{station}.{channel}.SAC")
                path_sm_st = os.path.join(path_sm, f"{network}.{station}.{channel}.SAC")

                st_sm = read(path_sm_st)
                tr_sm = st_sm[0]
                st_la = read(path_la_st)
                tr_la = st_la[0]
                d_az = tr_la.stats.sac.az
                df_la = tr_la.stats.sampling_rate
                df_sm = tr_sm.stats.sampling_rate
                tla0 = tr_la.stats.sac.t0 - tr_la.stats.sac.b
                tsm0 = tla0

                cc_all_max = []
                ltj_all = []
                ltjj_all = []
                ltc_all = []

                # Iterate over time shifts for large and small event traces
                for ltj in delta_tj:
                    for ltjj in delta_tjj:
                        tla = tla0 + ltj
                        tsm = tsm0 + ltjj + ltc

                        t0l = int(round(tla * df_la))
                        t1l = t0l + int(t_window * df_la)
                        t0s = int(round(tsm * df_sm))
                        t1s = t0s + int(t_window * df_sm)
                        time_cc = int(0.3 * df_la)
                        [cc_calc, cc_calc_scaled] = cc_cent_custom(
                            tr_la.data[t0l:t1l],
                            tr_sm.data[t0s:t1s],
                            time_cc,
                            time_cc
                        )
                        cc_all_max.append(cc_calc_scaled)
                        ltj_all.append(ltj)
                        ltjj_all.append(ltjj)
                        ltc_all.append(ltc)

                # Determine the maximum cross-correlation value and the corresponding time shifts
                cc_max1_list = cc_all_max.index(max(cc_all_max))
                ltj_max = ltj_all[cc_max1_list]
                ltjj_max = ltjj_all[cc_max1_list]
                ltc_max = ltc
                cc_inf.append([ltj_max, ltjj_max, ltc_max, max(cc_all_max), tla0, d_az])
                cc_max_station_all.append(max(cc_all_max))

            ccmax.append(np.mean(cc_max_station_all))
            ccmax_inf.append(cc_inf)

        cc_max_list = ccmax.index(max(ccmax))

        tla_all = []
        tsm_all = []
        ttj_all = []
        ttc_all = []
        ttjj_all = []
        cc = []
        st_az = []
        iv = 0
        for jj in li:
            network = st_network[jj]
            station = st_station[jj]
            channel = st_channel[jj]
            path_la_st = os.path.join(path_la, f"{network}.{station}.{channel}.SAC")
            st_la = read(path_la_st)
            tr_la = st_la[0]
            d_az = tr_la.stats.sac.az

            ltj_max = ccmax_inf[cc_max_list][jj][0]
            ltjj_max = ccmax_inf[cc_max_list][jj][1]
            ltc_max = ccmax_inf[cc_max_list][jj][2]
            cc_mmax = ccmax_inf[cc_max_list][jj][3]
            tla0 = ccmax_inf[cc_max_list][jj][4]

            tsm0 = tla0

            tla = tla0 + ltj_max
            tsm = tsm0 + ltjj_max + ltc_max

            tla_all.append(tla)
            tsm_all.append(tsm)
            ttj_all.append(ltj_max)
            ttc_all.append(ltc_max)
            ttjj_all.append(ltjj_max)
            cc.append(cc_mmax)
            st_az.append(d_az)
            iv += 1

        mean_cc = np.mean(cc)

        # Find event parameters for the large and small events
        (year_la, month_la, day_la, hour_la, min_la, sec_la, ev_lat_la, ev_lon_la,
         ev_dep_la, ev_mag_la) = find_eq(evid_all, lines_all, evid_la)
        (year_sm, month_sm, day_sm, hour_sm, min_sm, sec_sm, ev_lat_sm, ev_lon_sm,
         ev_dep_sm, ev_mag_sm) = find_eq(evid_all, lines_all, evid_sm)
        dy = ev_lat_la - ev_lat_sm
        dx = ev_lon_la - ev_lon_sm
        dz = ev_dep_la - ev_dep_sm
        dist_xy = math.sqrt(dx * dx + dy * dy) * 111
        dist_z = abs(dz)

        # Write cross-correlation and event info to output files
        f0.write(
            f"{evid_la} {evid_sm} {mean_cc} {year_la} {month_la} {day_la} {hour_la} {min_la} {sec_la} "
            f"{ev_lat_la} {ev_lon_la} {ev_dep_la} {ev_mag_la} {year_sm} {month_sm} {day_sm} {hour_sm} {min_sm} {sec_sm} "
            f"{ev_lat_sm} {ev_lon_sm} {ev_dep_sm} {ev_mag_sm} {st_az}\n"
        )
        fst.write(
            f"{evid_la} {evid_sm} {mean_cc} {st_network} {st_station} {st_channel} {st_az}\n"
        )

        # Plot maps if the mean cross-correlation exceeds certain thresholds
        if mean_cc > 0.8:
            plot_cc_map(
                st_network, st_station, st_channel, tla_all, tsm_all, cc, t_window,
                seg_t_start, seg_t_end, evid_la, evid_sm, path_la, path_sm, str_png
            )
        if mean_cc > 0.9:
            plot_cc_map(
                st_network, st_station, st_channel, tla_all, tsm_all, cc, t_window,
                seg_t_start, seg_t_end, evid_la, evid_sm, path_la, path_sm, str_png1
            )

# Close output files
f0.close()
fst.close()
