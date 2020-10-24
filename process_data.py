""" 
MIT License
Copyright (c) 2020 Yoga Suhas Kuruba Manjunath
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import shutil
import sys
import numpy as np
import pandas as pd
import pyunpack
import wget


def train_test_validation_set_split(x, y, train_ratio, test_ratio, validation_ratio):
    x_train, x_interim = np.split(x, [int(train_ratio *len(x))])
    y_train, y_interim = np.split(y, [int(train_ratio *len(y))])

    x_test, x_val = np.split(x_interim, [int(test_ratio *len(x))])
    y_test, y_val = np.split(y_interim, [int(test_ratio *len(y))])

    return x_train, x_test, x_val, y_train, y_test, y_val

def download_from_url(url, output_path):

    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')


def recreate_folder(path):
    """Deletes and recreates folder."""

    shutil.rmtree(path)
    os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
        url: Web address
        zip_path: Path to download zip file
        csv_path: Expected path to csv file
        data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')    


def download_traffic(force_download):
    """Downloads traffic dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip'

    data_folder = "data"
    csv_path = os.path.join(data_folder, 'PEMS_train')
    zip_path = os.path.join(data_folder, 'PEMS-SF.zip')

    if force_download:
        download_and_unzip(url, zip_path, csv_path, data_folder)

    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            l = [
                variable_type(i) for i in s.replace('[', '').replace(']', '').split()
            ]
        else:
            l = [
                variable_type(i)
                for i in s.replace('[', '').replace(']', '').split(delimiter)
            ]

        return l

    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), 'r') as dat:
            l = process_list(dat.readlines()[0])
        return l

    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(data_folder, filename), 'r') as dat:

            lines = dat.readlines()
            for i, line in enumerate(lines):
                array = [
                    process_list(row_split, variable_type=float, delimiter=None)
                    for row_split in process_list(
                        line, variable_type=str, delimiter=';')
                ]
                array_list.append(array)

        return array_list

    shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
    train_dayofweek = read_single_list('PEMS_trainlabels')
    train_tensor = read_matrix('PEMS_train')
    test_dayofweek = read_single_list('PEMS_testlabels')
    test_tensor = read_matrix('PEMS_test')

    inverse_mapping = {
        new_location: previous_location
        for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array([
        inverse_mapping[new_location]
        for new_location, _ in enumerate(shuffle_order)
    ])

    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = np.array(day_of_week[reverse_shuffle_order])
    combined_tensor = np.array(combined_tensor[reverse_shuffle_order])
  
    return combined_tensor, day_of_week


def load_data(data, force_download):

    if force_download:
        print('forceful download...')
        recreate_folder("data")    
    
    if data == "Traffic" or data == "traffic":
        print("loading data...")
        X , y = download_traffic(force_download)
        
    return X, y
    

if __name__ == "__main__":

    X, y = load_data(data = "Traffic", force_download = False)
    
    print(X[0])
    print(X[0].shape)
    print(y[0])
    print(y.shape)
