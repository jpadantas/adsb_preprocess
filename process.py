import argparse
import os
import csv
from glob import glob
import datetime
import pandas as pd
import numpy as np
import sys
from utils import get_runway_transform, convert_frame, get_date_time
from getWindVelocity import wind_params_runway_frame
import time


class Data:
    def __init__(self, data_path, weather_path):
        self.base_path = data_path
        self.data_path_raw = data_path + 'raw_data/'
        self.weather_path = weather_path
        self.out = 1
        self.last_wind_dir = 0
        self.last_wind_speed = 0
        self.last_h_km = 0
        self.last_h_km_debug = 0
        self.weather = None
        self.faa_dataset = None
        self.filelist = [y for x in os.walk(self.data_path_raw) for y in glob(os.path.join(x[0], '*.csv'))]
        self.R = get_runway_transform()
        self.alt_thold = 6000  # ft
        self.range_thold = 10  # km
        self.interpolation_size = 120
        self.define_memory_size()
        self.read_weather()
        self.read_faa_dataset()
        self.process_data()

    @staticmethod
    def define_memory_size():
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)

    def read_weather(self):

        self.weather = pd.read_csv(self.weather_path, low_memory=False)
        self.weather['datetime'] = pd.to_datetime(self.weather['valid'], format="%Y-%m-%d %H:%M")

    def read_faa_dataset(self):

        self.faa_dataset = pd.read_csv(os.path.dirname(os.getcwd()) + '/ReleasableAircraft/MASTER.txt',
                                       low_memory=False)

    def convert_to_local_df(self, df):
        # Converts data to local frame

        data = pd.DataFrame()
        data["datetime"] = df["Frame"]
        data['z'] = df.apply(lambda x: float(x["Altitude"]) * 0.3048 / 1000.0, axis=1)  # ft to km
        df["pos"] = df.apply(lambda x: convert_frame(float(x["Range"]), float(x["Bearing"]), self.R), axis=1)
        df[["x", "y"]] = pd.DataFrame(df.pos.tolist(), index=df.index)
        data['x'] = df.apply(lambda l: l.x[0], axis=1)
        data['y'] = df.apply(lambda l: l.y[0], axis=1)
        data['ID'] = df["ID"]

        return data

    def interpolate_data(self, df):

        # Interpolates the data

        df['datetime'] = pd.to_datetime(df['datetime'], format="%m/%d/%Y,%H:%M:%S")
        df.index = df['datetime']
        del df['datetime']
        # df.set_index('datetime', inplace=True)
        df_interpol = df.groupby('ID').resample('S').mean()
        df_interpol['x'] = df_interpol['x'].interpolate(limit=self.interpolation_size)
        df_interpol['y'] = df_interpol['y'].interpolate(limit=self.interpolation_size)
        df_interpol['z'] = df_interpol['z'].interpolate(limit=self.interpolation_size)
        del df_interpol['ID']
        df_interpol.reset_index(level=0, inplace=True)
        df_sorted = df_interpol.sort_values(by="datetime")
        df_sorted["time"] = df_sorted.index
        first = df_sorted["time"].iloc[0]
        df_sorted["Frame"] = (df_sorted["time"] - first).dt.total_seconds()
        df_sorted["Frame"] = df_sorted["Frame"].astype("int")
        del df_sorted["time"]
        df_sorted = df_sorted.dropna()

        return df_sorted

    def get_wind(self, utc):

        curr_utc = str(utc)
        utc_formatted = curr_utc[0:10] + "-" + curr_utc[11:-3]
        utc_time = datetime.datetime.strptime(utc_formatted, "%Y-%m-%d-%H:%M")

        result_index = self.weather['datetime'].sub(utc_time).abs().idxmin()
        try:
            wind_speed = float(self.weather["sknt"].iloc[result_index]) * 0.51444  # knots to m/s
            wind_angle = float(self.weather["drct"].iloc[result_index]) * np.pi / 180.0  # knots to m/s
            self.last_wind_dir = wind_angle
            self.last_wind_speed = wind_speed
        except ValueError:
            wind_angle = self.last_wind_dir
            wind_speed = self.last_wind_speed

        h_i, c_i = wind_params_runway_frame(wind_speed, wind_angle)
        if np.isnan(h_i):
            print("nan", h_i, utc)

        return h_i, c_i

    def get_pressure_alt(self, utc):
        curr_utc = str(utc)
        utc_formatted = curr_utc[0:10] + "-" + curr_utc[11:-3]
        utc_time = datetime.datetime.strptime(utc_formatted, "%Y-%m-%d-%H:%M")

        result_index = self.weather['datetime'].sub(utc_time).abs().idxmin()

        try:
            p_sta = float(self.weather["alti"].iloc[result_index]) * 33.8639  # inHg to mb
            h_alt = (1 - (p_sta / 1013.25) ** 0.190284) * 145366.45
            h_km = 0.3048 * h_alt * 1e-3
            self.last_h_km = h_km
        except ValueError:
            h_km = self.last_h_km
        return h_km

    def get_pressure_alt_debug(self, utc):
        curr_utc = str(utc)
        utc_formatted = curr_utc[0:10] + "-" + curr_utc[11:-3]
        utc_time = datetime.datetime.strptime(utc_formatted, "%Y-%m-%d-%H:%M")

        result_index = self.weather['datetime'].sub(utc_time).abs().idxmin()

        try:
            h_km = float(self.weather["alti"].iloc[result_index])
            self.last_h_km_debug = h_km
        except ValueError:
            h_km = self.last_h_km_debug
        return h_km

    def process_data(self):
        # Main loop: Reads each file
        for filename in self.filelist:
            print("Raw Filename =", filename)

            # Skip empty files
            if os.stat(filename).st_size == 0:
                print('file is empty')
                continue
            try:
                # Reading the file
                df = pd.read_csv(filename, engine='python', encoding='utf-8', delimiter=',', on_bad_lines='skip')

                # Dropping any NaN values
                df = df.dropna()

                # UNCOMMENT THIS PART IF YOU WANT TO FILTER USING THE FAA DATASET
                # Analysing each aircraft separately to filter the dataset
                # Proccess data filtering using the FAA Dataset
                # for tail in df['Tail'].dropna().unique():
                #
                #     aircraft_type = self.faa_dataset[self.faa_dataset['N-NUMBER'] == tail]['TYPE AIRCRAFT']
                #
                #     # Is the aircraft in the faa dataset?
                #     # if not len(aircraft_type):
                #     #     df = df[df['Tail'] != tail]
                #     #     continue
                #
                #     # Filtering aircraft type
                #     if aircraft_type.iloc[0] != '4' and aircraft_type.iloc[0] != '5':
                #         df = df[df['Tail'] != tail]
                #         continue

                # Filtering by Altitude and Range
                df = df[(df['Altitude'] < self.alt_thold) & (df['Range'] < self.range_thold)]

                if len(df) == 0:
                    continue

                # Converting the Frame column
                df["Frame"] = df.apply(get_date_time, axis=1)

                # Getting x, y, z coordinates from Beam, Range and Altitude
                df = self.convert_to_local_df(df)

                # Interpolation
                df_sorted = self.interpolate_data(df)

                # Wind information
                utc_timestamps = pd.DataFrame()
                df_sorted["utc"] = df_sorted.index
                df_sorted["wind"] = df_sorted.apply(lambda x: self.get_wind(x.utc), axis=1)
                utc_timestamps[["x", "y"]] = pd.DataFrame(df_sorted.wind.tolist(), index=df_sorted.index)
                df_sorted['Headwind'] = utc_timestamps.apply(lambda l: l.x, axis=1)
                df_sorted['Crosswind'] = utc_timestamps.apply(lambda l: l.y, axis=1)

                # Pressure Altitude correction
                df_sorted["h_km"] = df_sorted.apply(lambda x: self.get_pressure_alt(x.utc), axis=1)
                df_sorted["z"] = df_sorted["z"] - df_sorted["h_km"]

                df_sorted = df_sorted.drop(["utc", "wind", "h_km"], axis=1)

                self.seg_and_save(df_sorted)
            except:
                print("\n\n\n\n\n\nFILE WITH ISSUES! ERROR!\n\n\n\n\n\n")

    def seg_and_save(self, df):

        # Segregates the data into scenes and saves
        path_to_save = self.base_path + "processed_data/"

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        filename = path_to_save + str(self.out) + ".txt"

        print("Processed Filename =", filename)

        file = open(filename, 'w')
        csv_writer = csv.DictWriter(file, fieldnames=["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"])
        first_time = int(df.iloc[0]["Frame"])
        for index, row in df.iterrows():
            last_time = int(row["Frame"])
            if not ((last_time - first_time) > 1):
                row_write = row.to_dict()
                csv_writer.writerow(row_write)
            else:
                file.close()
                self.out += 1
                filename = self.base_path + "processed_data/" + str(self.out) + ".txt"
                print("Filename = ", filename)
                file = open(filename, 'w')
                csv_writer = csv.DictWriter(file, fieldnames=["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"])
            first_time = last_time
        self.out += 1
        file.close()

    # Fix interpolation problems
    # def seg_and_save_2(self, df):
    #
    #     # Segregates the data into scenes and saves
    #     path_to_save = self.base_path + "/processed_data/"
    #
    #     if not os.path.exists(path_to_save):
    #         os.makedirs(path_to_save)
    #
    #     filename = path_to_save + str(self.out) + ".txt"
    #
    #     print("Filename = ", filename)
    #
    #     df.reset_index(inplace=True, drop=True)
    #     df = df[["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"]]
    #
    #     first_time = int(df.iloc[0]["Frame"])
    #     checkpoint = 0
    #     for index, row in df.iterrows():
    #         last_time = int(row["Frame"])
    #         if not ((last_time - first_time) > 1):
    #             pass
    #         else:
    #             df.drop('Frame', axis=1).iloc[checkpoint:index - self.interpolation_size].to_csv(filename, index=False,
    #                                                                                              header=False)
    #             self.out += 1
    #             filename = path_to_save + str(self.out) + ".txt"
    #             checkpoint = index
    #         if index == len(df) - 1:
    #             df.drop('Frame', axis=1).iloc[checkpoint:index].to_csv(filename, index=False, header=False)
    #             self.out += 1
    #             filename = path_to_save + str(self.out) + ".txt"
    #         first_time = last_time


if __name__ == '__main__':
    # starting time
    start = time.time()

    # Dataset params
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='test/')
    parser.add_argument('--weather_folder', type=str, default='weather_data')

    args = parser.parse_args()

    # data_path = os.path.dirname(os.getcwd()) + args.dataset_folder + args.dataset_name
    data_path = os.getcwd() + args.dataset_folder + args.dataset_name

    print("Processing data from ", data_path)

    # weather_path = os.path.dirname(os.getcwd()) + args.dataset_folder + args.weather_folder + "/weather.csv"
    weather_path = os.getcwd() + args.dataset_folder + args.weather_folder + "/weather.csv"

    data = Data(data_path, weather_path)

    # end time
    end = time.time()

    # total time taken
    print(f"\nRuntime of the program is {end - start}")
