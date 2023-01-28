import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
from mpl_toolkits.mplot3d import axes3d
import os
from glob import glob
from os.path import join, abspath
from os import getcwd, stat
import pandas as pd
import random
import imageio
from scipy.interpolate import InterpolatedUnivariateSpline


def plot_steps(df):
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [], []

    sc = ax.scatter(x, y, marker='o', alpha=1, c='r', s=3)
    # plt.xlim(df.Lon.min() - 0.1, df.Lon.max() + 0.1)
    # plt.ylim(df.Lat.min() - 0.1, df.Lat.max() + 0.1)
    plt.xlim(-80.30, -79.50)
    plt.ylim(40.2, 40.5)
    plt.grid(linestyle='--')
    ax.set_title('Plotting Data from KAGC Airport')
    ax.scatter(df.iloc[0]['Lon'], df.iloc[0]['Lat'], marker='*', c='b', s=30)

    # https://flightaware.com/resources/airport/KAGC/runway/10/28
    ax.plot(np.linspace(-79.9410996, -79.9177762, 100), np.linspace(40.3542842, 40.3543109, 100),
            color="black", lw=10, alpha=1, zorder=0)
    ax.plot(np.linspace(-79.9410996, -79.9177762, 100), np.linspace(40.3542842, 40.3543109, 100),
            '--', color="white", lw=1.5, alpha=1, zorder=0)
    ax.plot(np.linspace(-79.9343298, -79.9224336, 100), np.linspace(40.3572940, 40.3520574, 100),
            color="gray", lw=10, alpha=1, zorder=0)
    ax.plot(np.linspace(-79.9343298, -79.9224336, 100), np.linspace(40.3572940, 40.3520574, 100),
            '--', color="white", lw=1.5, alpha=1, zorder=0)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.draw()

    for i in range(len(df)):
        if i == 0 or i == len(df) - 1:
            continue
        x.append(df.iloc[i]['Lon'])
        y.append(df.iloc[i]['Lat'])
        sc.set_offsets(np.c_[x, y])
        fig.canvas.draw_idle()
        plt.pause(0.1)
    ax.scatter(df.iloc[len(df) - 1]['Lon'], df.iloc[len(df) - 1]['Lat'], marker='*', alpha=1, c='g', s=30)

    plt.waitforbuttonpress()


def plot_lat_lon(df):
    fig, ax = plt.subplots()
    ax.grid(linestyle='--', zorder=0)

    ax.scatter(df['Lon'].values, df['Lat'].values, marker='o', alpha=1, c='r', s=3)
    # plt.xlim(df.Lon.min() - 0.1, df.Lon.max() + 0.1)
    # plt.ylim(df.Lat.min() - 0.1, df.Lat.max() + 0.1)
    plt.xlim(-80.30, -79.50)
    plt.ylim(40.2, 40.5)
    plt.grid(linestyle='--')
    ax.set_title('Plotting Data from KAGC Airport')
    ax.scatter(df.iloc[0]['Lon'], df.iloc[0]['Lat'], marker='*', c='b', s=30)
    ax.scatter(df.iloc[len(df) - 1]['Lon'], df.iloc[len(df) - 1]['Lat'], marker='*', alpha=1, c='g', s=30)
    # https://flightaware.com/resources/airport/KAGC/runway/10/28
    ax.plot(np.linspace(-79.9410996, -79.9177762, 100), np.linspace(40.3542842, 40.3543109, 100),
            color="black", lw=10, alpha=1, zorder=0)
    ax.plot(np.linspace(-79.9410996, -79.9177762, 100), np.linspace(40.3542842, 40.3543109, 100),
            '--', color="white", lw=1.5, alpha=1, zorder=0)
    # ax.plot(np.linspace(-79.9343298, -79.9224336, 100), np.linspace(40.3572940, 40.3520574, 100),
    #         color="gray", lw=10, alpha=1, zorder=0)
    # ax.plot(np.linspace(-79.9343298, -79.9224336, 100), np.linspace(40.3572940, 40.3520574, 100),
    #         '--', color="white", lw=1.5, alpha=1, zorder=0)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()


def plot2d(x, y, id, counter, p1, p2, p3, p4):
    fig, ax = plt.subplots()
    ax.grid(linestyle='--', zorder=0)

    # print('p1[0]', p1[0])
    # print('p2[0]', p2[0])
    # print('p1[1]', p1[1])
    # print('p2[1]', p2[1])
    # print('p3[0]', p3[0])
    # print('p4[0]', p4[0])
    # print('p3[1]', p3[1])
    # print('p4[1]', p4[1])

    ax.plot(np.linspace(p1[0], p2[0], 100), np.linspace(p1[1], p2[1], 100), color="black", lw=10, alpha=1, zorder=1)
    ax.plot(np.linspace(p1[0], p2[0], 100), np.linspace(p1[1], p2[1], 100), '--', color="white", lw=2, alpha=1,
            zorder=1)

    ax.plot(np.linspace(p3[0], p4[0], 100), np.linspace(p3[1], p4[1], 100), color="gray", lw=10, alpha=1, zorder=1)
    ax.plot(np.linspace(p3[0], p4[0], 100), np.linspace(p3[1], p4[1], 100), '--', color="white", lw=2, alpha=1,
            zorder=1)

    ax.plot(x, y, color="red")

    ax.scatter(x[0], y[0], marker="*", color="blue", s=30, zorder=2)
    ax.scatter(x[-1], y[-1], marker="*", color="green", s=30, zorder=2)

    # ax.set_title(f'{id}')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_xlim(-5, 6)
    ax.set_ylim(-5, 5)

    path_to_save = '/home/jdantas/Documents/cmu/data/kagc/dataset/plots/'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    plt.savefig(path_to_save + f'{counter}.png')
    plt.close(fig)


def plot(x, y, z):
    x_grid = np.linspace(-11, 11, 10 * len(x))
    y_grid = np.linspace(-11, 11, 10 * len(y))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = np.zeros((10 * x.size, 10 * z.size))

    spline = sp.interpolate.Rbf(x, y, z, function='thin_plate', smooth=5, episilon=5)

    Z = spline(B1, B2)
    fig = plt.figure(figsize=(10, 6))
    ax = axes3d.Axes3D(fig)
    ax.plot_wireframe(B1, B2, Z)
    ax.plot_surface(B1, B2, Z, alpha=0.2)
    ax.scatter3D(x, y, z, c='r')
    ax.show()


def plot_all_2d():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(linestyle='--', zorder=0)

    ax.plot(np.linspace(0, 1.98134561, 100), np.linspace(0, 4.4408921e-16, 100), color="black", lw=10, alpha=1,
            zorder=1)
    ax.plot(np.linspace(0, 1.98134561, 100), np.linspace(0, 4.4408921e-16, 100), '--', color="white", lw=2, alpha=1,
            zorder=1)

    ax.plot(np.linspace(0.57561845, 1.5853412, 100), np.linspace(0.33329789, -0.2496813, 100), color="gray", lw=10,
            alpha=1, zorder=1)
    ax.plot(np.linspace(0.57561845, 1.5853412, 100), np.linspace(0.33329789, -0.2496813, 100), '--', color="white",
            lw=2, alpha=1,
            zorder=1)

    counter = 0
    full_path = join(abspath(getcwd()), '../dataset/59_days_10_km/processed_data/', "*.txt")
    files = glob(full_path)
    random.shuffle(files)

    for file_name in files:
        if counter > len(files) * 0.1:
            break
        if stat(file_name).st_size == 0:
            continue

        csv_reader = pd.read_csv(file_name, header=None, names=["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"],
                                 engine='python')
        for aircraft in csv_reader['ID'].dropna().unique():
            df_temp = csv_reader[csv_reader['ID'] == aircraft]
            x = df_temp['x'].values.flatten()
            y = df_temp['y'].values.flatten()
            ax.plot(x, y, color="red", alpha=0.3, zorder=0)
            counter += 1

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_xlim(-4, 7)
    ax.set_ylim(-5, 5)

    path_to_save = f'../plots/plot_all/'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    plt.savefig(path_to_save + 'plot.png')


def plot_frame_2d(version):
    counter = 0
    full_path = join(abspath(getcwd()), f'../dataset/{version}/processed_data/', "*.txt")
    files = glob(full_path)
    # random.shuffle(files)

    for file_name in files:
        # if counter > len(files) * 0.1:
        # if counter > 0:
        #     break
        if stat(file_name).st_size == 0:
            continue

        csv_reader = pd.read_csv(file_name, header=None, names=["Frame", "ID", "x", "y", "z", "Headwind", "Crosswind"],
                                 engine='python')
        for aircraft in csv_reader['ID'].dropna().unique():
            df_temp = csv_reader[csv_reader['ID'] == aircraft]
            df_temp.drop_duplicates(subset=['x', 'y', 'z'], keep='first', inplace=True)
            filenames = []

            for frame in range(len(df_temp)):
                x = df_temp.iloc[:frame]['x'].values.flatten()
                y = df_temp.iloc[:frame]['y'].values.flatten()

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.grid(linestyle='--', zorder=0)

                ax.plot(np.linspace(0, 1.98134561, 100), np.linspace(0, 4.4408921e-16, 100), color="black", lw=10,
                        alpha=1,
                        zorder=1)
                ax.plot(np.linspace(0, 1.98134561, 100), np.linspace(0, 4.4408921e-16, 100), '--', color="white", lw=2,
                        alpha=1,
                        zorder=1)

                ax.plot(np.linspace(0.57561845, 1.5853412, 100), np.linspace(0.33329789, -0.2496813, 100), color="gray",
                        lw=10,
                        alpha=1, zorder=1)
                ax.plot(np.linspace(0.57561845, 1.5853412, 100), np.linspace(0.33329789, -0.2496813, 100), '--',
                        color="white",
                        lw=2, alpha=1,
                        zorder=1)
                ax.plot(x, y, color="red", alpha=0.8, zorder=0)
                ax.set_xlabel('x (km)')
                ax.set_ylabel('y (km)')
                ax.set_xlim(-4, 7)
                ax.set_ylim(-5, 5)

                if len(x) != 0:
                    ax.scatter(x[-1], y[-1], color="blue", alpha=1, zorder=1, s=10)

                path_to_save = f'../plots/plot_frames/{counter}/'

                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)

                plot_name = path_to_save + f'{frame}.png'
                filenames.append(plot_name)

                # save frame
                plt.savefig(plot_name)
                plt.close()

            # build gif
            create_gif(filenames, counter, version)
            print(counter)
            counter += 1


def create_gif(filenames, counter, version):
    path_to_save = f'../plots/gifs/{version}/'

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    # Build gif
    with imageio.get_writer(path_to_save + f'{counter}.gif', mode='I', duration=0.08) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)


def plot_interpol(x, y, z):
    #.rcParams['legend.fontsize'] = 10

    # let's take only 20 points for original data:
    n = len(x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z, label='rough curve')

    # this variable represents distance along the curve:
    t = np.arange(n)

    # now let's refine it to 100 points:
    t2 = np.linspace(t.min(), t.max(), 1000)

    # interpolate vector components separately:
    x2 = InterpolatedUnivariateSpline(t, x)(t2)
    y2 = InterpolatedUnivariateSpline(t, y)(t2)
    z2 = InterpolatedUnivariateSpline(t, z)(t2)

    ax.plot(x2, y2, z2, label='interpolated curve')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    # plot_all_2d()
    plot_frame_2d(version='linear')
