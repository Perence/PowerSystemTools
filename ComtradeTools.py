import csv
import datetime as dt
import os
import time as tm

import numpy as np
import pandas as pd
import h5py as h5

import FileTools.FileTools as FT
import PSCADTools.PSCADOutTools as POT


def normalize(data_list, rangemin=0, rangemax=4096):
    x = np.array(data_list)  # returns a numpy array
    # y = ax+b format for linear conversion
    a = (x.min() - x.max()) / (rangemin - rangemax)
    if a == 0:
        b = 0
        y_scaled = x
    else:
        b = x.max() - a * rangemax
        y_scaled = (x - b) / a
    return y_scaled.astype(int), a, b


def create_cfg(case_file, scale, output_directory, rename_file, start_t='2019-01-01 000000', title='py',
               norm=True, form='ASCII'):
    # This function makes all data signals analog
    # To strictly follow comtrade standard, file must be renamed to 8 characters in length
    '''
    station_name,rec_dev_id,rev_year <CR/LF>
    TT,##A,##D <CR/LF>
    An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS <CR/LF>
    An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS <CR/LF>
    An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS <CR/LF>
    An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS <CR/LF>
    Dn,ch_id,ph,ccbm,y <CR/LF>
    Dn,ch_id,ph,ccbm,y <CR/LF>
    lf <CR/LF>
    nrates <CR/LF>
    samp,endsamp <CR/LF>
    samp,endsamp <CR/LF>
    dd/mm/yyyy,hh:mm:ss.ssssss <CR/LF>
    dd/mm/yyyy,hh:mm:ss.ssssss <CR/LF>
    ft <CR/LF>
    timemult <CR/LF>
    '''

    cfg_file_path = output_directory
    station_name = rename_file + '_' + title
    file_name = rename_file
    extension = case_file.Extension
    if extension == '.hd5':
        channels_network = POT.hdf5_to_channels(case_file.Path)
    elif extension == '.inf':
        channels_network = POT.get_channels(case_file)  # getting all channels from the file and store in array
    elif extension == '.mat':

        # Read in matlab file

        f = h5.File('somefile.mat', 'r')
        data = f.get('data/variable1')
        # convert to channels format as above
        pass
    else:
        print(extension + ' is not a correct file type. Skipping: ' + case_file.Directory)
        return

    # set station_name, rec_dev_id, and rev_year
    rec_dev_id = case_file.File_name_prefix
    rev_year = '1999'

    if not os.path.exists(cfg_file_path):
        os.makedirs(cfg_file_path)

    cfg_file = open(cfg_file_path + '//' + file_name + '.cfg', 'w', newline="")
    first_line = station_name, rec_dev_id, rev_year
    csv.writer(cfg_file).writerow(first_line)

    # %%
    # parse the array data to determine number of signals - analog and digital
    # assuming all signals are analog

    total_channels = str(len(channels_network))
    analogs = total_channels + 'A'
    digitals = '0D'

    second_line = total_channels, analogs, digitals
    csv.writer(cfg_file).writerow(second_line)

    # %%
    # Determine the start time (dd/mm/yyyy,hh:mm:ss.ssssss)
    start = channels_network[1].data.time[0]
    milisecond = "000000"
    second1 = start_t[-2:]
    start_second = str(int(start) + int(second1))
    minute = start_t[-4:-2]
    hour = start_t[-6:-4]
    day = start_t[-9:-7]
    month = start_t[-12:-10]
    year = start_t[-17:-13]
    date = day + "/" + month + "/" + year

    start_time = hour.rjust(2, '0') + ":" + minute.rjust(2, '0') + ":" + start_second.rjust(2, '0') + "." + milisecond

    # Determine the end time (dd/mm/yyyy,hh:mm:ss.ssssss)
    end = channels_network[1].data.time[-1]

    # %%
    # for each channel, set channel number, channel label, channel units, channel multiplier, channel offset, min, max,
    # primary, secondary, PS
    # Channel format (An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS)

    for channel in channels_network:
        chan_num = channel.channel_number
        chan_label = channel.description
        chan_phase = 'ph'
        chan_phase = ''
        #circ_monitor = 'NA'
        circ_monitor = ''
        #chan_units = 'unit'
        chan_units = channel.units
        #chan_units = ''
        if norm:
            set_min = 0
            set_max = 4096
            norm_chan, a, b = normalize(channel.data.data, set_min, set_max)
            chan_multiplier = np.float32(a)
            chan_offset = np.float32(b)
            chan_min = np.float32(min(norm_chan))
            chan_max = np.float32(max(norm_chan))
        else:
            chan_multiplier = 1
            chan_offset = 0
            chan_min = np.float32(min(channel.data.data))
            chan_max = np.float32(max(channel.data.data))
        time_skew = 0
        chan_prim = 1
        chan_sec = 1
        chan_ps = 'S'

        next_line = chan_num, chan_label, chan_phase, circ_monitor, chan_units, chan_multiplier, chan_offset, \
                    time_skew, chan_min, chan_max, chan_prim, chan_sec, chan_ps
        csv.writer(cfg_file).writerow(next_line)

    # %%
    # Set the line frequency - assume europe 50hz
    l_f = str(50.0)

    # Set the number of sampling rates in the file - assume 1
    nrates = str(1)

    # set the last sample in the file

    end_sample = int((len(channels_network[1].data.data) - 1) / scale)

    # Set sample rate in hz

    total_time = end - start

    sample_rate = end_sample / total_time

    # Set as ASCII file type
    # %%
    # Writing the end of the .cfg file

    sample_line = ['{0:.2f}'.format(sample_rate), '{1}'.format(sample_rate, end_sample)]
    start_string = date, start_time
    format_line = form
    time_mult = 1

    csv.writer(cfg_file).writerow([l_f])
    csv.writer(cfg_file).writerow(nrates)
    csv.writer(cfg_file).writerow(sample_line)
    csv.writer(cfg_file).writerow(start_string)
    csv.writer(cfg_file).writerow(start_string)  # set the trigger point to the same as the start
    csv.writer(cfg_file).writerow([format_line])
    csv.writer(cfg_file).writerow([time_mult])

    cfg_file.close()


def create_dat(case_file, scale, output_directory, rename_file, title='py', norm=True, form='ASCII'):
    dat_file_path = output_directory
    if rename_file is not '':
        file_name = rename_file
    else:
        file_name = case_file.File_name_prefix + '_' + title
    extension = case_file.Extension
    if extension == '.hd5':
        channels_network = POT.hdf5_to_channels(case_file.Path)
    elif extension == '.inf':
        channels_network = POT.get_channels(case_file)  # getting all channels from the file and store in array
    else:
        print(extension + ' is not a correct file type. Skipping: ' + case_file.Directory)
        return

    dat_file = FT.CommonFile(dat_file_path, file_name, False, ".dat", False)

    if not os.path.exists(dat_file_path):
        os.makedirs(dat_file_path)

    data = pd.DataFrame()
    rows = len(channels_network[0].data.data) - 1
    end_sample = int(rows / scale)
    scale_series = [x * scale for x in range(0, end_sample)]
    scaled_rows = len(scale_series)
    start = channels_network[1].data.time[0]
    end = channels_network[1].data.time[-1]
    total_time = end - start
    sample_freq = int(total_time / end_sample * 1e6) #in microseconds

    for channel in channels_network:
        if norm:
            norm_chan, chan_multiplier, chan_offset = normalize(channel.data.data)
            channel_data = pd.DataFrame(norm_chan, dtype=int)
        else:
            channel_data = pd.DataFrame(channel.data.data, dtype=np.float16)
        data = pd.concat([data, channel_data], axis=1)
    output = data.iloc[scale_series]
    output.insert(0, "RowNum", range(1, scaled_rows + 1))
    #output.insert(1, "Time", '')
    output.insert(1, "Time", range(0, scaled_rows * sample_freq, sample_freq))
    if form is 'ASCII':
        output.to_csv(dat_file.Path, index=False, header=False)
    elif form is 'Binary':
        output.to_hdf(dat_file.Path, index=False, header=False)
    else:
        print(form + ' is and invalid comtrade data format. No data file created.')



def get_time():
    dtnow = dt.datetime.fromtimestamp(tm.time())
    hour = dtnow.hour
    minute = dtnow.minute
    second = dtnow.second
    timestr = str(hour).rjust(2, '0') + ":" + str(minute).rjust(2, '0') + ":" + str(second).rjust(2, '0')
    return timestr


def get_cfg_channels(cfg_dir, file_name):
    def get_dat_file(cfgfile):
        return FT.CommonFile(cfgfile.Directory, cfgfile.File_name_prefix, False, ".dat",
                             False)

    def get_dat_file_df(cfgfile):
        file = get_dat_file(cfgfile)
        return pd.read_csv(file.Path, skiprows=0, header=None)

    filepath = cfg_dir + "\\" + file_name + ".cfg"
    f = open(filepath)

    cfg_file_contents = []
    channel_headers = []
    # read data in from cfg file

    for line in csv.reader(f):
        cfg_file_contents.append(line)
    f.close()

    cfg_channels = cfg_file_contents[2:-6]
    chan_num = 0
    chan_time_scl = 1.0e-6
    # 1999 - An,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS
    # 1991 - An,ch_id,ph,ccbm,uu,a,b,skew,min,max
    # get channel information and write headers
    for channel in cfg_channels:
        chan_num = chan_num + 1
        # comtrade data
        if len(channel) > 5:
            # analogue
            chan_mult = float(channel[5])
            chan_offset = float(channel[6])
            chan_min = float(channel[-2]) + float(chan_offset)
            chan_max = float(channel[-1]) * float(chan_mult) + float(chan_offset)

            # channel header: no, desc, group, max, min, units, multip factor, offset, timestep
            channel_headers.append(POT.ChannelHeader(chan_num, channel[1], cfg_file_contents[0][0],
                                                     chan_max, chan_min, "", chan_mult, chan_offset, chan_time_scl))
        else:
            # binary
            # channel header: Dn,ch_id,ph,ccbm,y
            channel_headers.append(POT.ChannelHeader(chan_num, channel[1], cfg_file_contents[0][0],
                                                     1, 0, "", ))

    cfg_file = FT.CommonFile(cfg_dir, file_name, False, ".cfg", False)
    df = get_dat_file_df(cfg_file)
    channels = []

    for channel_header in channel_headers:
        col_index = channel_header.channel_number
        channels.append(channel_header.to_channel(POT.OutputData(time=list(df[1]),
                                                                 data=list(df[col_index + 1]))))
    return channels


def cfg_to_hd5(data_dir, filename, start=None):
    channels_network = get_cfg_channels(data_dir,
                                        filename)  # getting all channels from the .cfg file and store in array
    POT.resize_channels(channels_network, start, None)  # truncate channels by removing up to 'start' time
    POT.channels_to_hdf5(channels_network, data_dir + "\\" + filename + "_datconv.hd5", float_type="float32")


def combine_pscad_cfg(files_dir, files_list):  # this assumes there are no digital signals
    # get title line and footer lines from first cfg file
    first_cfg_contents = FT.read_delimited_contents(files_dir, files_list[0], ".cfg")
    title_line = [first_cfg_contents[0][0] + "_combined_cfgs", first_cfg_contents[0][1]]
    footer = first_cfg_contents[-6:]

    # open new file for writing title it with the folder name + combined
    parent = files_dir.split("\\")[-2]
    new_files_dir = files_dir + "\\" + parent + "_COMBINED"
    if not os.path.exists(new_files_dir):
        os.makedirs(new_files_dir)
    new_cfg_name = parent + "_COMBINED"
    new_cfg_file = FT.CommonFile(new_files_dir, new_cfg_name, False, ".cfg", False)
    new_cfg = open(new_cfg_file.Path, "w", newline="")

    csv.writer(new_cfg).writerow(title_line)

    chan_num = 0
    channels = []
    for cfg_file in files_list:
        print(str(cfg_file) + ".cfg")
        # open file and read contents
        f_contents = FT.read_delimited_contents(files_dir, cfg_file, ".cfg")

        # read each channel data and replace the channel number with the appropriate channel number
        for line in f_contents[2:-6]:
            chan_num = chan_num + 1
            temp_line = line
            temp_line[0] = chan_num
            channels.append(temp_line)

    # write second line data
    csv.writer(new_cfg).writerow([str(chan_num), str(chan_num) + "A", "0D"])
    # write channel data
    csv.writer(new_cfg).writerows(channels)
    # write footer data
    csv.writer(new_cfg).writerows(footer)
    new_cfg.close()
    return new_files_dir


def combine_pscad_dat(files_dir, files_list):
    # create new directory if it doesnt exist
    parent = files_dir.split("\\")[-2]
    new_files_dir = files_dir + "\\" + parent + "_COMBINED"
    if not os.path.exists(new_files_dir):
        os.makedirs(new_files_dir)
    new_dat_name = parent + "_COMBINED"
    new_dat_file = FT.CommonFile(new_files_dir, new_dat_name, False, ".dat", False)

    # write the first two columns
    first_dat_contents = pd.DataFrame(FT.read_delimited_contents(files_dir, files_list[0], ".dat"))
    channels = first_dat_contents.iloc[:, 0:2]

    for file in files_list:
        print(str(file) + ".dat")
        f_contents = pd.DataFrame(FT.read_delimited_contents(files_dir, file, ".dat"))
        f_channels = f_contents.iloc[:, 2:]
        channels = pd.concat([channels, f_channels], axis=1)

    channels.to_csv(new_dat_file.Path, header=False, index=False)


def create_dat_bin(case, scale, working_directory, filename, filetype='.hd5'):
    """
    dat_file = open(dat_file_path + "\\comtrade\\" + station_name + '.dat', 'wb', newline="")

    def get_data_row(channels, count, filename):
        line = [count, ""]
        for channel in channels:
            line.append(np.float16(channel.data.data[(count - 1) * scale]))
        csv.writer(filename).writerow(line)

    count = 0
    for row in channels_network[0].data.data:
        count = count + 1
        if count * scale >= len(channels_network[0].data.data):
            break
        get_data_row(channels_network, count, dat_file)

    dat_file.close()
    """
