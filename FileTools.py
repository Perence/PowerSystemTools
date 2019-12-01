#
# FileTools was created by Geoff Love at PSC Consulting UK Ltd.
# FileTools was updated by Perry Hofbauer at PSC Consulting UK Ltd.
#
# Feel free to use the module in your program. Don't forget that FileTools is licensed under the GNU version 3 license. For more info about this read the *COPYING.txt*.
#
# Usage
# ===============
# The classes and functions below are provided as is. Generally designed to support file types of file management.
# Script was developed for internal usage and is being shared with the community to aid in knowledge sharing.
# This script is not maintained, contains undescribed limitations, and may include errors.
# No specific examples are provided. For questions in application, contact the authors
# Use at own risk
#

import csv
import datetime as dt

import pandas as pd

standard_column_delimiter = ","


class CommonFile(object):
    def __init__(self, directory, file_name_prefix, include_date=True, file_name_suffix=".txt", open_file=True):
        if isinstance(directory, str):
            self.Directory = self.create_directory(directory)
        else:  # assume list of strings
            self.Directory = self.form_directory(directory)
        self.File_name_prefix = file_name_prefix
        self.File_name = self.form_file_name(file_name_prefix, include_date, file_name_suffix)
        self.Path = self.form_path(self.Directory, self.File_name)
        self.File = self.open_file_for_writing() if open_file else None
        self.Extension = file_name_suffix

    def form_file_name(self, file_name_prefix, include_date, file_name_suffix):
        date_string = get_date_string() if include_date else ""
        return file_name_prefix + date_string + file_name_suffix

    def form_directory(self, directory_list):
        output_directory = directory_list[0]
        for directory in directory_list[1:]:
            output_directory = self.form_path(output_directory, directory)
        return self.create_directory(output_directory)

    def create_directory(self, directory):
        return directory if directory[-1] == "\\" else directory + "\\"

    def form_path(self, directory, file_name):
        if directory[-1] == '\\':
            return directory + file_name
        else:
            return directory + '\\' + file_name

    def open_file_for_writing(self):
        return open(self.Path, mode='w+')

    def read_contents(self, strip=True):
        self.File = open(self.Path)
        contents = self.File.readlines()
        if strip:
            return [x.strip() for x in contents]
        else:
            return contents

    def close(self):
        self.File.close()


class LogFile(CommonFile):
    def __init__(self, directory, file_name_prefix, include_date=True, file_name_suffix=".log", open_file=True,
                 include_date_in_log=True, include_id=True, column_delimiter=standard_column_delimiter):
        CommonFile.__init__(self, directory, file_name_prefix, include_date, file_name_suffix, open_file)
        self.Include_Date_In_Log = include_date_in_log
        self.Include_ID = include_id
        self.Count = 0
        self.Column_Delimiter = column_delimiter

    def write_log(self, code, description):
        date = get_date_string() + self.Column_Delimiter if self.Include_Date_In_Log else ""
        if self.Include_ID:
            self.Count += self.Count
            row_id = str(self.Count) + self.Column_Delimiter
        else:
            row_id = ""
        print(date + row_id + code + description, file=self.File)


class PandasCsvFile(CommonFile):
    def __init__(self, directory, file_name_prefix, include_date=True, file_name_suffix=".csv", load_data_frame=True,
                 column_delimiter=standard_column_delimiter, number_of_col_headers=1):
        CommonFile.__init__(self, directory, file_name_prefix, include_date, file_name_suffix, False)
        self.Column_Delimiter = column_delimiter
        self.No_Col_Headers = number_of_col_headers
        self.DF = self.load_data_frame() if load_data_frame else None

    def load_data_frame(self):
        return pd.read_csv(self.Path)

    def save_data_frame(self):
        return self.DF.to_csv(self)


class PandasResultsExcelFile(CommonFile):
    def __init__(self, directory, file_name_prefix, file_name_suffix=".xlsx", include_date=False):
        CommonFile.__init__(self, directory, file_name_prefix, include_date, file_name_suffix, False)
        self.Writer = None

    def load_data_frame(self, sheetname, verbose=True):
        if verbose:
            print("loading sheet: " + sheetname + " from: " + self.File_name)
        return pd.read_excel(self.Path, header=[0, 1], sheetname=sheetname)

    def save_data_frame(self, data_frame, sheet):
        if self.Writer is None:
            self.Writer = pd.ExcelWriter(self.Path)
        data_frame.to_excel(self.Writer, str(sheet))

    def save_data(self, index, data, column_names, sheet_name=None):
        if sheet_name is None:
            sheet_name = column_names[0]
        if self.Writer is None:
            self.Writer = pd.ExcelWriter(self.Path)

        data_frame = pd.DataFrame(data, index, column_names)

        data_frame.to_excel(self.Writer, str(sheet_name))

    def save_writer(self):
        self.Writer.save()

    def sheet_names(self):
        return pd.ExcelFile(self.Path).sheet_names


class PandasExcelFile(CommonFile):
    def __init__(self, directory, file_name_prefix, sheetname=None, file_name_suffix=".xlsx", load_data_frame=True,
                 column_delimiter=standard_column_delimiter, include_date=False, number_of_col_headers=1):
        CommonFile.__init__(self, directory, file_name_prefix, include_date, file_name_suffix, False)
        self.Column_Delimiter = column_delimiter
        self.No_Col_Headers = number_of_col_headers
        self.SheetName = sheetname
        self.DF = self.load_data_frame() if load_data_frame else None
        self.Columns = self.DF.columns

    def get_row(self, index):
        return self.Df.loc[index, :]

    def filter_rows(self, filter_cols, filter_values):
        pairs = zip(filter_cols, filter_values)
        filtered_df = self.DF
        for pair in pairs:
            filtered_df = filtered_df[filtered_df[pair[0]] == pair[1]]
        return filtered_df

    def get_values(self, filter_cols, filter_values, value_col):
        return list(self.filter_rows(filter_cols, filter_values)[value_col])

    def load_data_frame(self):
        return pd.read_excel(self.Path, sheetname=self.SheetName)

    def save_data_frame(self):
        return self.DF.to_excel(self)


def get_date_string():
    return dt.datetime.now().strftime("%Y-%m-%d %H%M%S")


def get_file_type_list(dir, filetype):
    list = []
    for file in dir:
        if filetype in file:
            list.append(file.replace(filetype, ""))
    return list


def read_delimited_contents(file_dir, filename, filetype):
    file_obj = CommonFile(file_dir, filename, False, filetype, False)
    file = open(file_obj.Path)
    file_contents = []
    for line in csv.reader(file):
        file_contents.append(line)
    file.close()
    return file_contents