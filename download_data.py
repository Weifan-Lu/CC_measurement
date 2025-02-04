import os
import sys
import re
import math
import glob
import datetime
import subprocess
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy import read, UTCDateTime, read_inventory
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees

# Uncomment if needed:
# from obspy.clients.fdsn import Client
# import multiprocessing as mp

from function_download import event_waveform, event_waveform_sacpz, event_waveform_inf, pick_phase, load_gf

# Function to call SAC command
def a_change(fpath, kt0, outpath, fillz=False):
    p = subprocess.Popen(['sac'], stdin=subprocess.PIPE)
    s = "wild echo off \n"
    # Uncomment the following line if you need to fill zeros
    # if fillz: s += "cuterr fillz \n"
    s += "cut -5 40\n"
    s += "r %s \n" % (fpath)
    s += "ch a %s\n " % (kt0)
    s += "w %s \n" % (outpath)
    s += "q \n"
    p.communicate(s.encode())

# Function to calculate SNR for a single channel using RMS values
def SNR_singlech_rms(data_seis, tp, t_before, t_after, len_s, len_n):
    S = data_seis

    if len(S) < 5000:
        snr = 1000
    else:
        S_signal = S[tp + t_after: tp + t_after + len_s]
        S_noise = S[tp - t_before - len_n: tp - t_before]
        if len(S_signal) < 10 or len(S_noise) < 10:
            snr = 1000
        else:
            PS = math.sqrt(sum([x ** 2 for x in S_signal]) / len(S_signal))
            PN = math.sqrt(sum([x ** 2 for x in S_noise]) / len(S_noise))
            if PS == 0:
                snr = 1000
            else:
                snr = PN / PS
    return snr

# Input file and path configurations
fevent = '../Download_data/large_all_event.txt'
# fevent = '../Download_data/small_all_event.txt'

# Read the event file
with open(fevent, 'r') as f:
    lines = f.readlines()

str_front_phase = '../Download_data/phase_data/phase/'
st_pha = '../Events_la/'
cha = "*HZ"
catalog = "NCSS"

# Adjust the following indices and numbers as needed
ilo = 121
ilo_line = ilo - 1
num_j = 122
progress_bar = True
snr_vel = 0.1
t_start = 5
t_end = 80

# Iterate through event lines
for line in lines:
    if ilo <= num_j:
        ilo_line += 1
        line = lines[ilo_line]
        ilo += 1

        # Parse event information
        parts = line.split()
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        sec = float(parts[5])
        sec1 = int(sec)
        sec2 = round((sec - sec1) * 1000)
        evid0 = int(parts[10])
        ev_lat = float(parts[6])
        ev_lon = float(parts[7])
        ev_dep = float(parts[8])
        ev_mag = float(parts[9])
        event_path = os.path.join(st_pha, str(evid0))
        t0 = UTCDateTime(year, month, day, hour, minute, sec)
        fsmall_ph = os.path.join(str_front_phase, str(evid0) + '.dat')

        if os.path.exists(event_path):
            print('Event directory already exists, skipping...')
        else:
            if os.path.exists(fsmall_ph):
                # Create required directories
                os.mkdir(event_path)
                mseed_path = os.path.join(event_path, 'mseed')
                os.mkdir(mseed_path)
                sacpz_path = os.path.join(event_path, 'sacpz')
                os.mkdir(sacpz_path)
                bkdata_path = os.path.join(event_path, 'bkdata')
                os.mkdir(bkdata_path)
                pzdata_path = os.path.join(event_path, 'pzdata')
                os.mkdir(pzdata_path)

                out_file = os.path.join(mseed_path, str(evid0) + '.mseed')
                # Download waveform data
                event_waveform(evid0, catalog, cha, t0 - t_start, t0 + t_end, out_file, progress_bar=True)
                size = os.path.getsize(out_file)
                if size > 10:
                    print("This station's data size is not zero.")
                    st_all = read(out_file)
                    with open(fsmall_ph, 'r') as fph:
                        linesph = fph.readlines()
                    for sj in range(len(st_all)):
                        st = st_all[sj]
                        network1 = st.stats.network
                        station1 = st.stats.station
                        channel1 = st.stats.channel
                        location1 = st.stats.location
                        if location1 == '':
                            location1 = '--'

                        # Pick the phase based on station info and event time
                        [p_arrival, p_label] = pick_phase(linesph, station1, channel1, network1, t0)
                        if p_arrival > 0 and p_label == "P":
                            # Save the SAC format data to bkdata
                            out_file_sac = os.path.join(bkdata_path, "{}.{}.{}.sac".format(network1, station1, channel1))
                            st.write(out_file_sac, format='SAC')
                            # Download SAC pole-zero (sacpz) data
                            out_file_sacpz = os.path.join(sacpz_path, "SACPZ.{}.{}.{}.{}".format(network1, station1, '', channel1))
                            event_waveform_sacpz(network1, station1, location1, channel1, t0 - 20, t0 + 100, out_file_sacpz, progress_bar=True)
                            size_sacpz = os.path.getsize(out_file_sacpz)
                            if size_sacpz > 10:
                                type_val, constant, st_lat, st_lon, st_el = load_gf(out_file_sacpz)
                                print(evid0, station1)
                                event_waveform_inf(out_file_sac, line, st_lat, st_lon, st_el)
                                st = read(out_file_sac)
                                st.detrend('demean').detrend('linear').taper(max_percentage=0.05)
                                tr = st[0]
                                delta = tr.stats.delta
                                p_time = p_arrival
                                len_s = 1
                                len_n = 1
                                t_before = 1.1
                                t_after = 0.1
                                snr = SNR_singlech_rms(
                                    tr.data,
                                    int((p_time + t_start) / delta),
                                    int(t_before / delta),
                                    int(t_after / delta),
                                    int(len_s / delta),
                                    int(len_n / delta)
                                )
                                if snr < snr_vel and constant > 0:
                                    s = ""
                                    s += "r {}/bkdata/{}.{}.{}.sac \n".format(event_path, network1, station1, channel1)
                                    s += "div {} \n".format(constant)
                                    s += "mul 1.0e9 \n"
                                    s += "ch idep IVEL\n"
                                    s += "ch t0 %s\n" % (p_time)
                                    s += "w  {}/pzdata/{}.{}.{}.SAC \n".format(event_path, network1, station1, channel1)
                                    s += "q \n"
                                    subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())

