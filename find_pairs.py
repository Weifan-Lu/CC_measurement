from obspy import read, UTCDateTime
import os
import sys
import shutil
import subprocess

# Append the function directory to the Python path


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Disable SAC copyright display
os.putenv("SAC_DISPLAY_COPYRIGHT", '0')

import warnings
warnings.filterwarnings("ignore")
from scipy import signal
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel
import math
from function_all import select_station



# Path to the event file containing links
fevent = '../Download_data/link_all.txt'

# Read all lines from the event file
with open(fevent, 'r') as f:
    lines = f.readlines()

# Define directories for events
str_front_la = '../Events_sm/'
str_front_sm = '../Events_sm/'

ixx = 1
num_j = 0  # Set the maximum number of events to process
# Alternatively, you can use:
# ilo = 0
# num_j = 10000000
# or
# ilo = 0
# num_j = 10000

t_window = 0.2       # Time window (seconds)
slope_num = 70       # Slope threshold (unit depends on your context)
seg_t_start = 60     # Segment start time (seconds)
seg_t_end = 180      # Segment end time (seconds)

# Create the output file for cross-correlation results
str_cc_st = 'file/link_' + str(0) + '_all_eqs.txt'
fst = open(str_cc_st, 'w')

snr_vl = 0.1  # SNR threshold

ilo = 0  # Initialize event counter

for ii in range(len(lines)):
    if ilo <= num_j:
        line = lines[ilo]
        ilo += 1
        print(line)
        # Parse event IDs from the line
        evid_la = int(line.split(' ')[0])
        evid_sm = int(line.split(' ')[1])
        path_la = os.path.join(str_front_la, str(evid_la), 'pzdata')
        path_sm = os.path.join(str_front_sm, str(evid_sm), 'pzdata')

        # Process only if both event paths exist
        if os.path.exists(path_la) and os.path.exists(path_sm):
            # Get a list of SAC files in the small event directory
            sac_list = [os.path.join(path_sm, i) for i in os.listdir(path_sm) if i.endswith('.SAC')]
            print(len(sac_list))
            if len(sac_list) > 9:
                read_sac = os.path.join(str_front_sm, str(evid_sm), 'pzdata', '*.SAC')
                st_sm = read(read_sac)
                # Select stations based on the small event stream, the large event path, and SNR threshold
                st_network, st_station, st_channel, st_distance = select_station(st_sm, path_la, snr_vl)
                li = list(range(len(st_distance)))
                ccmax = []
                ccmax_inf = []

                # Only write if the mean station distance is less than 10000 (units should be defined in your context)
                if np.mean(st_distance) < 10000:
                    fst.write(
                        f"{evid_la} {evid_sm} {st_network} {st_station} {st_channel}\n"
                    )

# Close the output file after processing
fst.close()

