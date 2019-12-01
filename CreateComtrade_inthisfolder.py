#
# CreateComtrade_inthisfolder was created by Perry Hofbauer at PSC Consulting UK Ltd.
#
# Feel free to use the module in your program. Don't forget that # CreateComtrade_inthisfolder is licensed under the GNU version 3 license. For more info about this read the *COPYING.txt*.
#
# Usage
# ===============
# The classes and functions below are provided as is. This script is an example for the conversion of PSCAD .out files to standard COMTRADE format for post-processing automation.
# Script was developed for internal usage and is being shared with the community to aid in knowledge sharing.
# This script is not maintained and no specific examples are provided. For questions in application, contact the author.
#
###
# This program is designed to take .hd5 or .inf/.out files and convert them into comtrade readable output
# To run this script set 3 variables:
# 1) the working directory (working_directory) - this is the main folder location of the files.
#       The script will work through subfolders
# 2) The file type (set_file) this is either '.hd5' or '.inf'
# 3) The scale factor (scale_factor) this reduces the overall filesize if required
###

import os
import time as tm

import FileTools.FileTools as FT
import PSCADTools.PSCADComtradeTools as CT

start = tm.time()

working_directory = r"C:\Users\..." 'Folder where .out and .inf file is stored'

output_dir = working_directory + '\\Comtrade'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
scale_factor = int(1)  # set this to reduce size of comtrade files

# create .cfg and .dat files
start_time = '2019-01-01 000000'

# find the name of the inf or hd5 file
file_names = [f for f in os.listdir(working_directory) if (f.endswith('.hd5') or f.endswith('.inf'))]
for file_name in file_names:
    extension = '.' + file_name.split('.')[-1]
    file_prefix = file_name.split('.')[-2]
    if (extension != '.inf') and (extension != '.hd5'):
        print(extension + ' is not a correct file type. Skipping: ' + working_directory + '\\' + file_name)
        continue
    network_file = FT.CommonFile(working_directory, file_prefix, False, extension, False)

    print('\n{s:+^100}'.format(s=' ' + file_prefix + ' '))
    print(" Creating .cfg file..." + CT.get_time())
    CT.create_cfg(network_file, scale_factor, output_dir, file_prefix, start_time)
    print(" Creating .dat file..." + CT.get_time())
    CT.create_dat(network_file, scale_factor, output_dir, file_prefix)

total_time = tm.time() - start
minute = total_time // 60
sec = total_time % 60
print('\nScript run took {0} minutes and {1:.2f} seconds'.format(int(minute), sec))
