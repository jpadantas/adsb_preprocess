

from geographiclib.geodesic import Geodesic
import datetime
import numpy as np
import math
from ast import literal_eval
import matplotlib.pyplot as plt


def get_date_time(row):
    ##Get datetime object
    date = [int(i) for i in literal_eval(row["Date"])]
    time = [int(math.floor(float(l))) for l in literal_eval(row["Time"])]
    t = datetime.datetime(date[0],date[1],date[2],time[0],time[1],time[2])
    return t


def get_range_and_bearing(lat1,lon1,lat2,lon2):
    ##Get relative range and bearing between two lat/lon points
    geod = Geodesic.WGS84
    lat2 = float(lat2)
    lon2 = float(lon2)
    g = geod.Inverse(lat1,lon1,lat2,lon2)

    return g['s12']/1000.0, g['azi1']

def get_runway_transform():
    ##Numbers are hardcoded to KBTP
    R1 = [40.774548, -79.959237] ##Runway 08
    R2 = [40.778630, -79.942803] ##Runway 26
    cam_ref = [ 40.777888, -79.949864] ##setup location
    runway_length = 1.45
    r1, b1 = get_range_and_bearing(cam_ref[0],cam_ref[1],R1[0],R1[1])
    x1 = r1*np.sin(np.deg2rad(b1))
    y1 = r1*np.cos(np.deg2rad(b1))
    # print(x1,y1)
    r2, b2 = get_range_and_bearing(cam_ref[0],cam_ref[1],R2[0],R2[1])
    x2 = r2*np.sin(np.deg2rad(b2))
    y2 = r2*np.cos(np.deg2rad(b2))
    # print(x2,y2)
    ang = -np.arctan2(y1-y2,x1-x2)
    rot = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    p = -rot@np.array([x1,y1])
    # print(p)
    R = -np.array([[np.cos(ang),-np.sin(ang), p[0]],[np.sin(ang),np.cos(ang),p[1]],[0,0,1]])
    
    return R

def convert_frame(r,b,R):
    x = np.multiply(r,np.sin(np.deg2rad(b)))
    y = np.multiply(r,np.cos(np.deg2rad(b)))
    points = np.matmul(R,np.vstack((x,y,np.ones((np.shape(x))))))
    return points[0],points[1]

def feet2kmeter(value):

    return float(value)*0.3048/1000.0

def get_pressure_alt_inverted(h_km):
    h_alt = h_km / (0.3048 * 1e-3)
    p_sta = 1013.25 * ((1 - (h_alt / 145366.45)) ** (1 / 0.190284))
    return p_sta / 33.8639


def plot2d(x, y):
    fig, ax = plt.subplots()
    ax.grid(linestyle='--', zorder=0)

    ax.plot(np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), color="black", lw=10, alpha=1, zorder=0)
    ax.plot(np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), '--', color="white", lw=2, alpha=1, zorder=0)

    ax.plot(x, y, color="red")

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)

    ax.axis('equal')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    # ax.view_init(azim=90, elev=90)

    plt.show()


def plot2d_all(df):
    fig, ax = plt.subplots()
    ax.grid(linestyle='--', zorder=0)
    ax.plot(np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), color="black", lw=10, alpha=1, zorder=0)
    ax.plot(np.linspace(0, 1.45, 100), np.linspace(0, 0, 100), '--', color="white", lw=2, alpha=1, zorder=0)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.axis('equal')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'pink', 'brown', 'black', 'gray', 'purple', 'orange', 'olive']
    c = 0
    for id in df['ID'].unique():

        df_temp = df[df['ID'] == id]

        ax.plot(df_temp['x (km)'].values.flatten(), df_temp['y (km)'].values.flatten(), color=colors[c])
        c += 1

    plt.show()


def calculate_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

if __name__ == '__main__':

    R1 = [40.778208, -79.968666]
    # R1 = [ 40.777888, -79.949864]
    cam_ref = [ 40.777888, -79.949864]
    r1, b1 = get_range_and_bearing(cam_ref[0],cam_ref[1],R1[0],R1[1])
    R = get_runway_transform()
    print(convert_frame(r1,b1,R))