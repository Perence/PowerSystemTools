#
# PSCADOutTools was created by Geoff Love at PSC Consulting UK Ltd.
# PSCADOutTools was updated by Perry Hofbauer at PSC Consulting UK Ltd.
#
#
# Feel free to use the module in your program. Don't forget that PSCADOutTools is licensed under the GNU version 3 license. For more info about this read the *COPYING.txt*.
#
# Usage
# ===============
# The classes and functions below are provided as is. Generally designed to support working with PSCAD .out files for post-processing automation.
# Script was developed for internal usage and is being shared with the community to aid in knowledge sharing.
# This script is not maintained, contains undescribed limitations, and may include errors.
# No specific examples are provided. For questions in application, contact the authors
# Use at own risk
#

import pandas as pd
import math as m
import numpy as np
import cmath as cm
import FileTools.FileTools as FT
import collections as c
import fnmatch as fnm

class ChannelHeader(object):
    def __init__(self, channel_number, Desc, Group, Max, Min, Units, Mult=1, Offset=0, TimeScale=1):
        self.channel_number = channel_number
        self.description = Desc
        self.group = Group
        self.max = Max
        self.min = Min
        self.units = Units
        self.full_name = self.group + ":" + self.description
        self.mult = Mult
        self.offset = Offset
        self.timescale = TimeScale


    def to_channel(self, data):
        return Channel(self.channel_number, self.description, self.group, self.max, self.min, self.units, data,
                       self.mult, self.offset, self.timescale)

    def add_group_prefix(self, prefix):
        self.group = self.group + prefix

    def rename_channel(self, new_decription, new_group=None):
        self.description = new_decription
        if new_group:
            self.group = new_group


class OutputData(object):
    def __init__(self, time, data):
        self.time = time
        self.data = data

    def get_time(self, t):
        if t == -1:
            return self.data[-1]
        else:
            index = self.time.index(t)
            return self.data[index]


def find_channel(channels, description, group=None):
    return next(channel for channel in channels if channel.description == description)


def contains_channel(channels, description, group=None):
    channel = next((channel for channel in channels if channel.description == description),None)
    if channel:
        return True
    else:
        return False


def append_prefix_to_channels(channels, prefix):
    [channel.add_group_prefix(prefix) for channel in channels]


def get_dictionary_of_channels(channels, t):
    dictionary = {}
    for channel in channels:
        dictionary[channel.full_name] = channel.get_time(t)
    return dictionary


class Channel(ChannelHeader):
    def __init__(self, channel_number, Desc, Group, Max, Min, Units, data, Mult=1, Offset=0, TimeScale=1):
        ChannelHeader.__init__(self, channel_number, Desc, Group, Max, Min, Units, Mult, Offset, TimeScale)
        if self.timescale != 1:
            data.time = [self.timescale*x for x in data.time]
        if (self.mult != 1) or (self.offset != 0):
            data.data = [self.mult*x+self.offset for x in data.data]
        self.data = data

    def get_time(self, t):
        return self.data.get_time(t)

    def get_time_index(self, t):
        return min(range(len(self.data.time)), key=lambda i: abs(self.data.time[i] - t))

    def get_final_time(self):
        return self.data.time[-1]

    def get_time_data(self):
        return self.data.time, self.data.data

    def resize_data(self, t_min_index, t_max_index):
        self.data.time = self.data.time[t_min_index: t_max_index]
        self.data.data = self.data.data[t_min_index: t_max_index]

    def time_shift(self, t_shift):
        self.data.time = [t + t_shift for t in self.data.time]

    def correct_zeroes(self):
        for idx, data in enumerate(self.data.data):
            if abs(data) < 1e-7:
                self.data.data[idx] = 0

    def __repr__(self):
        return self.group + ":" + self.description


def scale_channel(channel, scale):
    channel_new = channel
    channel_new.data.data = [scale*(data) for data in channel_new.data.data]
    return channel


def get_channels(inf_file):
    file_contents = inf_file.read_contents()
    channel_headers = []

    for line in file_contents:

        def find(s, ch):
            return [i for i, ltr in enumerate(s) if ltr == ch]

        def replace_spaces_in_speach_marks(line):
            indices = find(line, "\"")
            lead_indices = indices[0::2]
            trail_indices = indices[1::2]
            index_pairs = list(zip(lead_indices, trail_indices))
            for pair in index_pairs:
                sub_string = line[pair[0] + 1:pair[1]]
                sub_string = sub_string.replace(" ", "_")
                line = line[0:pair[0] + 1] + sub_string + line[pair[1]:]
            return line

        def get_channel_number(string_with_chanel_number):
            first_bracket = string_with_chanel_number.index('(')
            second_bracket = string_with_chanel_number.index(')')
            return int(string_with_chanel_number[first_bracket + 1:second_bracket])

        def get_kwags(string_with_kwags):
            kwags_start = 23

            return dict([(k, v) for k, v in (pair.split('=') for pair in string_with_kwags[kwags_start:]
                        .replace("\"", "").split())])

        channel_number = get_channel_number(line)
        line = replace_spaces_in_speach_marks(line)
        kwags = get_kwags(line)

        channel_headers.append(ChannelHeader(channel_number, **kwags))

    number_of_out_files = determine_out_file_number(channel_headers[-1])[0]
    out_file_range = range(1, number_of_out_files + 1)

    data_frames = {}

    for out_number in out_file_range:
        data_frames[out_number] = get_out_file_df(inf_file, out_number)

    channels = []
    for channel_header in channel_headers:
        out_file_num, col_index = determine_out_file_number(channel_header)
        df = data_frames[out_file_num]
        channels.append(channel_header.to_channel(OutputData(time=list(df[0]), data=list(df[col_index]))))

    return channels


def channels_to_data_frame(channels, float_type=None):
    time = channels[0].data.time
    channel_numbers = []
    descs = []
    groups = []
    maxs = []
    mins = []
    units = []
    data = []

    # self.channel_number = channel_number
    # self.description = Desc
    # self.group = Group
    # self.max = Max
    # self.min = Min
    # self.units = Units

    for channel in channels:
        data.append(channel.data.data)
        channel_numbers.append(channel.channel_number)
        descs.append(channel.description)
        groups.append(channel.group)
        maxs.append(channel.max)
        mins.append(channel.min)
        units.append(channel.units)

    header = pd.MultiIndex.from_arrays([channel_numbers, descs, groups, maxs, mins, units],
                                       names=["channel_number", "description", "group", "max", "min", "units"])

    data = list(map(list, zip(*data)))
    if float_type == "float32":
        df = pd.DataFrame(data, columns=header, index=time, dtype=np.float32)
    elif float_type == "float16":
        df = pd.DataFrame(data, columns=header, index=time, dtype=np.float16)
    else:
        df = pd.DataFrame(data, columns=header, index=time)
    return df


def sub_channel(channel, time_min=None, time_max=None):
    if time_min and time_max:
        channel.data.data = channel.data.data[channel.get_time_index(time_min):channel.get_time_index(time_max)]
        channel.data.time = channel.data.time[channel.get_time_index(time_min):channel.get_time_index(time_max)]
    elif time_min and time_max is None:
        channel.data.data = channel.data.data[channel.get_time_index(time_min):]
        channel.data.time = channel.data.time[channel.get_time_index(time_min):]
    elif time_min is None and time_max:
        channel.data.data = channel.data.data[:channel.get_time_index(time_max)]
        channel.data.time = channel.data.time[:channel.get_time_index(time_max)]
    return channel


def resize_channels(channels, time_min=None, time_max=None):
    # assumes all channels have the same time data

    if time_min:
        t_min_index = channels[0].get_time_index(time_min)
    else:
        t_min_index = None

    if time_max:
        t_max_index = channels[0].get_time_index(time_max)
    else:
        t_max_index = None

    [channel.resize_data(t_min_index, t_max_index) for channel in channels]


def get_all_channels(channels, channel_filter, filter_by_description=True):
    if filter_by_description:
        return [channel for channel in channels if fnm.fnmatch(channel.description, channel_filter)]
    else:
        return [channel for channel in channels if fnm.fnmatch(channel.full_name, channel_filter)]


def df_to_channels(data_frame):
    time = data_frame.index
    return [Channel(*column[0:6], OutputData(time, data_frame[column].tolist()), *column[6:]) for column in data_frame]


def channels_to_hdf5(channels, hdf5_file, key="df", float_type=None, compression_type="bzip2"):
    df = channels_to_data_frame(channels, float_type)
    df.to_hdf(hdf5_file, key=key, complib=compression_type)


def hdf5_to_channels(hdf5_file, key="df"):

    if type(hdf5_file) is str:
        df = pd.read_hdf(hdf5_file, key)
    elif type(hdf5_file) is FT.CommonFile:
        df = pd.read_hdf(hdf5_file.Path, key)
    else:
        raise TypeError(hdf5_file)
    return df_to_channels(df)


def get_channels_from_group(channels, group_name):
    group_name = group_name.replace(" ", "_")
    try:
        return [channel for channel in channels if channel.group == group_name]
    except StopIteration as error:
        return None


def get_channel(channels, channel_name):
    channel_name = channel_name.replace(" ", "_")
    try:
        return next(channel for channel in channels if channel.description == channel_name)
    except StopIteration as error:
        return None


def calculate_specturm(channel, t_step=None, N=None):
    if t_step is None:
        t_step = channel.data.time[-1] - channel.data.time[-2]

    if N is None:
        N = int(1 / t_step)

    # sp = scipy.fftpack.fft(channel.data.data[-N-1:-1])*2/N
    # freq = scipy.fftpack.fftfreq(len(sp))*N
    freq = np.fft.rfftfreq(len(channel.data.time[-N - 1:-1]), t_step)
    sp = 2 / N * np.fft.rfft(channel.data.data[-N - 1:-1])
    return freq, sp


def calculate_specturm_as_mags(channel, t_step=None, N=None, f0=1):
    freq, polar = calculate_specturm_as_polar(channel, t_step, N, f0)
    return freq, [x[0] for x in polar]


def calculate_specturm_as_polar(channel, t_step=None, N=None, f0=1):
    freq, sp = freq, sp = calculate_specturm(channel, t_step, N)
    freq = [harm / f0 for harm in freq]
    polar = [cm.polar(ele) for ele in sp]
    return freq, polar


def calculate_specturm_as_dic_mags(channel, t_step=None, N=None):
    freq, sp = freq, sp = calculate_specturm(channel, t_step, N)
    sp_mag = [abs(ele) for ele in sp]
    return c.OrderedDict(sorted(dict(zip(freq, sp_mag)).items()))


def calculate_specturm_as_dic(channel, t_step=None, N=None):
    freq, sp = calculate_specturm(channel, t_step, N)
    return c.OrderedDict(sorted(dict(zip(freq, sp)).items()))


def calculate_impedance_specturm_as_dic(v_channel, i_channel, t_step=None, N=None):
    if t_step is None:
        t_step = v_channel.data.time[-1] - v_channel.data.time[-2]

    if N is None:
        N = int(1 / t_step)

    freq, sp_va = calculate_specturm(v_channel, t_step, N)
    freq, sp_ia = calculate_specturm(i_channel, t_step, N)

    sp_z = np.array([va / ia for (va, ia) in zip(sp_va, sp_ia)])

    return c.OrderedDict(sorted(dict(zip(freq, sp_z)).items()))


def determine_out_file_number(channel):
    max_column_number = 10
    return m.ceil(channel.channel_number / max_column_number), \
           max_column_number \
               if channel.channel_number % max_column_number == 0 \
               else channel.channel_number % (max_column_number)


def get_out_file(inf_file, out_number):
    return FT.CommonFile(inf_file.Directory, inf_file.File_name_prefix + "_" + str(out_number).zfill(2), False, ".out",
                         False)


def get_out_file_df(inf_file, out_number):
    file = get_out_file(inf_file, out_number)
    return pd.read_csv(file.Path, skiprows=1, header=None, delim_whitespace=True)


def calc_3phase_rms(channels, channel_filter):
    inst_channels = get_all_channels(channels, '*' + channel_filter + '*', False)
    rms = []
    for time_index, data in enumerate(inst_channels[0].data.data):
        rms.append((np.sqrt(np.mean(data**2) + np.mean(inst_channels[1].data.data[time_index]**2)
                           + np.mean(inst_channels[2].data.data[time_index]**2)))/np.sqrt(2))
    rms_channel = Channel(len(channels), channel_filter + '_RMS', inst_channels[0].group, max(rms), min(rms), inst_channels[0].units, rms)
    return rms_channel