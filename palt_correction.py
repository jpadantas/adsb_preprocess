from glob import glob
from os.path import join, abspath
from os import getcwd, stat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

columns = ['Frame#', 'AircraftID', 'x (km)', 'y (km)', 'z (km)', 'windx (m/s)', 'windy (m/s)',
           'press alt factor (km)']
full_path = join(abspath(getcwd()), '../dataset/154_days_10_km_palt/processed_data', "*.txt")


def apply_correction():
    for file_name in glob(full_path):
        if stat(file_name).st_size == 0:
            continue

        csv_reader = pd.read_csv(file_name, header=None, delimiter=' ', names=columns)

        csv_reader['z (km)'] = csv_reader['z (km)'] - csv_reader['press alt factor (km)']
        csv_reader.drop('press alt factor (km)', axis=1, inplace=True)
        csv_reader = csv_reader.drop[0]
        csv_reader.to_csv(file_name, index=False)


def apply_correction_header():
    for file_name in glob(full_path):
        if stat(file_name).st_size == 0:
            continue

        csv_reader = pd.read_csv(file_name, header=None)

        csv_reader = csv_reader.iloc[1:, :].reset_index(drop=True)
        csv_reader.to_csv(file_name, header=False, index=False)


def plot2d():
    for file_name in glob(full_path):
        if stat(file_name).st_size == 0:
            continue

        csv_reader = pd.read_csv(file_name)

        fig, ax = plt.subplots()
        ax.grid(linestyle='--', zorder=0)

        ax.plot(np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), color="black", lw=10, alpha=1, zorder=0)
        ax.plot(np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), '--', color="white", lw=2, alpha=1, zorder=0)

        ax.plot(csv_reader['x (km)'].values.flatten(), csv_reader['y (km)'].values.flatten(), color="red")

        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)

        plt.show()
        plt.close(fig)


def create_dataset():
    df = pd.DataFrame(columns=columns[:-1])
    for file in glob(full_path):
        if stat(file).st_size == 0:
            continue

        csv_file = pd.read_csv(file)

        df = pd.concat([df, csv_file], axis=0, ignore_index=True)
    df.to_csv('/home/jdantas/Documents/cmu/data/dataframes/df_154_10_km_palt.csv', index=False)


if __name__ == '__main__':
    apply_correction_header()
