from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import matplotlib.patches as patches
import subprocess
from scipy import signal
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import math
import os
from matplotlib.patches import Ellipse, Circle



def cut(fpath, b, e, kt0, outpath, fillz=False):
    p = subprocess.Popen(['sac'], stdin=subprocess.PIPE)
    s = "wild echo off \n"
    s += "cut %s %s \n" %(b, e)
    s += "r %s \n" %(fpath)
    s += "ch a %s\n "%(kt0)
    s += "w %s \n" %(outpath)
    s += "q \n"
    p.communicate(s.encode())


def select_station(st_sm, path_la, snr_vel):
    st_network = []
    st_station = []
    st_channel = []
    st_distance = []
    iv = 0
    st_evid = []
    st_inf = []
    li = list(range(len(st_sm)))
    my_array = np.empty((len(st_sm), 2))

    for jj in li:
        tr_sm = st_sm[jj]
        df = tr_sm.stats.sampling_rate
        network = tr_sm.stats.network
        station = tr_sm.stats.station
        channel = tr_sm.stats.channel
        path_la_one = path_la + str(network) + '.' + str(station) + '.' + str(channel) + '.SAC'
        st_network.append(network)
        st_station.append(station)
        st_channel.append(channel)

        if channel == 'EHZ' or channel == 'HHZ':
            if os.path.exists(path_la_one):
                st_la = read(path_la_one)
                tr_la = st_la[0]
                tr_la.data = tr_la.data - np.mean(tr_la.data)
                tr_sm.data = tr_sm.data - np.mean(tr_sm.data)

                p_tla = tr_la.stats.sac.t0 - tr_la.stats.sac.b
                p_tsm = tr_sm.stats.sac.t0 - tr_sm.stats.sac.b
                if p_tla > 0 and p_tsm > 0:
                    # Parameters for SNR and slope (now computed but not used for decision-making)
                    len_s = 1
                    len_n = 1
                    t_before = 1.1
                    t_after = 0.1
                    p_time = tr_la.stats.sac.t0 - tr_la.stats.sac.b
                    delta = tr_la.stats.sampling_rate

                    if p_time < 2.1:
                        len_n = p_time / 2
                        t_before = p_time / 4

                    snr1, PN1 = SNR_singlech_rms(tr_la.data, int(p_time * delta), int(t_before * delta),
                                                 int(t_after * delta), int(len_s * delta), int(len_n * delta))
                    slope33 = slope(tr_la.data, int(p_time * delta), 2, 2)

                    # Repeat similar parameters for the small event trace
                    len_s = 1
                    len_n = 1
                    t_before = 1.1
                    t_after = 0.1
                    p_time = tr_sm.stats.sac.t0 - tr_sm.stats.sac.b
                    delta = tr_sm.stats.sampling_rate
                    if p_time < 2.1:
                        len_n = p_time / 2
                        t_before = p_time / 4
                    snr2, PN2 = SNR_singlech_rms(tr_sm.data, int(p_time * delta), int(t_before * delta),
                                                 int(t_after * delta), int(len_s * delta), int(len_n * delta))
                    slope44 = slope(tr_sm.data, int(p_time * delta), 2, 2)
                    slope_max = slope33 * slope44

                    print(snr1, snr2, len_n, t_before)
                    # Instead of checking SNR and slope values, always set the distance as follows:
                    distance = tr_la.stats.sac.dist / 1000
                else:
                    distance = 1000000
            else:
                # If the corresponding large event SAC file does not exist
                snr1 = 1000
                snr2 = 1000
                distance = 1000000000
        else:
            distance = 1000000

        st_distance.append(float(distance))
        st_evid.append(iv)
        st_inf.append([iv, network, station, channel, distance])
        my_array[jj, 0] = iv
        my_array[jj, 1] = distance
        iv += 1

    # Sort stations based on the calculated distance
    my_array = my_array[np.argsort(my_array[:, 1])]

    pick_network = []
    pick_station = []
    pick_channel = []
    pick_distance = []
    st_len = len(st_station) if len(st_station) < 10 else 10

    for sj in range(st_len):
        evidd = int(my_array[sj][0])
        pick_network.append(st_network[evidd])
        pick_station.append(st_station[evidd])
        pick_channel.append(st_channel[evidd])
        pick_distance.append(st_distance[evidd])

    st_network = pick_network
    st_station = pick_station
    st_channel = pick_channel
    st_distance = pick_distance
    return st_network, st_station, st_channel, st_distance

def cc_cent_custom(waveone, wavetwo, nptccln, nptwave):
    # Calculates cross-correlation between 2 waveforms

    cc_calc = np.zeros((int(nptccln - nptwave + 1), 1))
    cc_calc_scaled = np.zeros((int(nptccln - nptwave + 1), 1))

    for idt in range(int(nptccln - nptwave + 1)):
        cctmp1 = (waveone - np.mean(waveone))
        cctmp2 = (wavetwo[idt:idt + nptwave] - np.mean(wavetwo[idt:idt + nptwave]))
        if len(cctmp1) == len(cctmp2):
            amp1 = np.sqrt(np.dot(cctmp1, cctmp1))
            amp2 = np.sqrt(np.dot(cctmp2, cctmp2))
            cc_scaled = np.maximum(amp1*amp1 , amp2*amp2)
            if amp1 != 0 and amp2 != 0:
                cc_calc[idt] = np.dot(cctmp1, cctmp2) / amp1 / amp2
                cc_calc_scaled[idt] = np.dot(cctmp1, cctmp2) / cc_scaled
            else:
                cc_calc[idt] = 0
        else:
            cc_calc[idt] = 0

    return cc_calc,cc_calc_scaled




def plot_cc_map(st_network, st_station, st_channel, tla_all, tsm_all, cc, t_window, seg_t_start, seg_t_end, evid_la,
                    evid_sm, path_la, path_sm, str_file):

        mean_cc = np.mean(cc)

        fig = plt.figure(figsize=(20, 10))
        l_la_sm = st_network
        li = [i for i in range(len(l_la_sm))]
        lii = [i for i in range(len(l_la_sm) + 1)]
        sub_len = len(l_la_sm) / 3
        if isinstance(sub_len, int) == True:
            sub_len = sub_len
        else:
            sub_len = int(sub_len) + 1

        for il in lii:
            if il == 0:
                ax = plt.subplot2grid((4, 4), (0, 0), colspan=1)
            elif il == 1:
                ax = plt.subplot2grid((4, 4), (0, 1), colspan=1)
            elif il == 2:
                ax = plt.subplot2grid((4, 4), (1, 0), colspan=1)
            elif il == 3:
                ax = plt.subplot2grid((4, 4), (1, 1), colspan=1)
            elif il == 4:
                ax = plt.subplot2grid((4, 4), (2, 0), colspan=1)
            elif il == 5:
                ax = plt.subplot2grid((4, 4), (2, 1), colspan=1)
            elif il == 6:
                ax = plt.subplot2grid((4, 4), (3, 0), colspan=1)
            elif il == 7:
                ax = plt.subplot2grid((4, 4), (3, 1), colspan=1)
            elif il == 8:
                ax = plt.subplot2grid((4, 4), (3, 2), colspan=1)
            elif il == 9:
                ax = plt.subplot2grid((4, 4), (3, 3), colspan=1)
            elif il == 10:
                ax = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=3)

            if il < 10:
                network = st_network[il]
                station = st_station[il]
                channel = st_channel[il]
                path_la_st = path_la + str(network) + '.' + str(station) + '.' + str(channel) + '.SAC'
                path_sm_st = path_sm + str(network) + '.' + str(station) + '.' + str(channel) + '.SAC'

                st_sm = read(path_sm_st)
                st_la = read(path_la_st)

                tla0 = st_la[0].stats.sac.t0 - st_la[0].stats.sac.b
                tsm0 = tla0

                tla_seis = [i for i in range(st_la[0].stats.npts)]
                tla_seis = np.array(tla_seis)
                dfla = st_la[0].stats.sampling_rate
                tla_seis = tla_seis / dfla - tla0

                tsm_seis = [i for i in range(st_sm[0].stats.npts)]
                tsm_seis = np.array(tsm_seis)
                dfsm = st_sm[0].stats.sampling_rate
                tsm_seis = tsm_seis / dfsm - tsm0

                sm_label = '   Small Mag: ' + str(st_sm[0].stats.sac.mag) + '  StartTime: ' + str(
                    st_sm[0].stats.starttime - st_sm[0].stats.sac.b)
                la_label = '   Large Mag: ' + str(st_la[0].stats.sac.mag) + '  StartTime: ' + str(
                    st_la[0].stats.starttime - st_la[0].stats.sac.b)
                if il == 0 or il == 2 or il == 4 or il == 6:
                    ax.set_ylabel('Velocity (' + chr(956) + 'm/s)')
                if il == 6 or il == 7 or il == 8 or il == 9:
                    ax.set_xlabel('Time (s)')

                x0l = (tla_all[il] + st_la[0].stats.sac.b - st_la[0].stats.sac.t0) - 0.01
                x0s = (tsm_all[il] + st_la[0].stats.sac.b - st_la[0].stats.sac.t0) - 0.01
                diff_ls = x0l - x0s
                r_ymax = max(st_la[0].data) * 1.2
                ax.add_patch(patches.Rectangle((x0l, -r_ymax), t_window, r_ymax * 2, facecolor="gray", alpha=0.3))

                st_la_data = st_la[0].data
                st_sm_data = st_sm[0].data
                st_la_data = st_la_data - st_la_data[int(tla_all[il]*dfla)]
                st_sm_data = st_sm_data - st_sm_data[int(tsm_all[il]*dfsm)]



                lime_y = max(abs(st_sm_data)) / 1e3

                ax.plot(tla_seis, st_la_data / 1e3, c='black', markersize=15, label=la_label)
                ax.plot(tsm_seis + diff_ls, st_sm_data / 1e3, c='red', markersize=15, label=sm_label)
                str_cc = str(np.around(cc[il], 3)).replace("[","")
                str_cc = str_cc.replace("]","")
                cc_label = 'CC: ' + str_cc

                station0_label = str(st_la[0].stats.network) + '.' + str(st_la[0].stats.station) +'.' + str(st_la[0].stats.channel)
                ax.text(-0.18, lime_y * 0.6, cc_label)
                ax.text(-0.18, lime_y * 0.8, station0_label)

                ax.set_xlim(-0.2, 0.8)
                ax.set_ylim(-lime_y * 1.2, lime_y * 1.2)

            else:
                st_dist = []
                for il in li:
                    network = st_network[il]
                    station = st_station[il]
                    channel = st_channel[il]
                    path_la_st = path_la + str(network) + '.' + str(station) + '.' + str(channel) + '.SAC'
                    path_sm_st = path_sm + str(network) + '.' + str(station) + '.' + str(channel) + '.SAC'

                    st_sm = read(path_sm_st)
                    tr_sm = st_sm[0]
                    st_la = read(path_la_st)
                    tr_la = st_la[0]
                    network1 = st_sm[0].stats.network
                    station1 = st_sm[0].stats.station
                    channel1 = st_sm[0].stats.channel
                    st_dist.append(st_sm[0].stats.sac.dist/1000/111)

                    lat = st_sm[0].stats.sac.stla
                    lon = st_sm[0].stats.sac.stlo
                    ax.plot(lon, lat, '^', c='blue', markersize=15)
                    str_station = str(network1) + '.' + str(station1)
                    ax.text(lon + 0.005, lat, str_station)
                    radius_la = 20 / 111
                    cir1 = Circle(xy=(st_la[0].stats.sac.evlo, st_la[0].stats.sac.evla), radius=radius_la, color='gray',
                                  fill=False)
                    ax.add_patch(cir1)
                    ax.set_aspect("equal", adjustable="box")

                ep_dist = np.max(st_dist) * 1.2
                sm_label = '   Small Mag ' + str(st_sm[0].stats.sac.mag) + '  StartTime: ' + str(
                    st_sm[0].stats.starttime - st_sm[0].stats.sac.b)
                la_label = '   Large Mag: ' + str(st_la[0].stats.sac.mag) + '  StartTime: ' + str(
                    st_la[0].stats.starttime - st_la[0].stats.sac.b)
                plt.plot(st_sm[0].stats.sac.evlo, st_sm[0].stats.sac.evla, '.', c='red', markersize=15,
                         label='Small EQ')
                plt.plot(st_la[0].stats.sac.evlo, st_la[0].stats.sac.evla, '.', c='black', markersize=15,
                         label='Large EQ')
                plt.xlim([st_la[0].stats.sac.evlo - ep_dist, st_la[0].stats.sac.evlo + ep_dist])
                plt.ylim([st_la[0].stats.sac.evla - ep_dist, st_la[0].stats.sac.evla + ep_dist])
                plt.legend(loc='upper left')
                plt.suptitle(str(round(mean_cc, 3)) + sm_label + la_label)

                fig_link = str_file + 'CCmax_' + str(evid_la) + '_' + str(evid_sm) + '_event_waveform.png'
                plt.savefig(fig_link, dpi=300)


def find_eq(evid_all,lines_all,evid_la):
    position = evid_all.index(evid_la)
    line = lines_all[position]
    year = int(line.split(' ')[0])
    month = int(line.split(' ')[1])
    day = int(line.split(' ')[2])
    hour = int(line.split(' ')[3])
    min = int(line.split(' ')[4])
    sec = float(line.split(' ')[5])
    sec1 = int(sec)
    sec2 = round((sec - sec1) * 1000)
    evid0 = int(line.split(' ')[10])
    ev_lat = float(line.split(' ')[6])
    ev_lon = float(line.split(' ')[7])
    ev_dep = float(line.split(' ')[8])
    ev_mag = float(line.split(' ')[9])
    return year,month,day,hour,min,sec,ev_lat,ev_lon,ev_dep,ev_mag