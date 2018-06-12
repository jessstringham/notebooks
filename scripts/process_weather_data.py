'''Goes with the post from 2018-06-11.
Hacky script for extracting a (hopefully) random subset of weather data.
I'm using this for a tiny side project, and I haven't verified it totally
works, so be careful if you're doing anything serious with it!

This takes the un-tar'd download from

ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_gsn.tar.gz

or

ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_hcn.tar.gz

and writes a file to DESTINATION_NPZ. It tries to extract TRAIN_EXAMPLES_PER_STATION
examples per station, and will remove the station if there are fewer than
MIN_COUNT_PER_STATION examples for the station.
Each example contains the station_id, month, day, and values for each of
INCLUDED_ELEMENTS.

It also writes a file STATIONS_FILE, where the line index is the station_id
used in DESTINATION_NPZ, and the contents of the line are the station name
used in the other GHCN data, including ghcnd-stations.txt.

It throws out data that is low quality or missing.

This is preprocessed with a script which removes data from before 1990.

    #!/bin/bash


    SRC_FOLDER=$1
    DEST_FOLDER=$2

    year_awk_program='{
    if ( substr($0, 12, 4) > 1990 )
      print $0
    }'


    for filename in ${SRC_FOLDER}/*.dly; do
        echo 'before'
        wc -l $filename
        cat $filename | awk "$year_awk_program" > "${filename}-tmp"
        echo 'after'
        wc -l "${filename}-tmp"
    done

    mkdir -p $DEST_FOLDER

    mv ${SRC_FOLDER}/*.dly-tmp ${DEST_FOLDER}

    # and remove the -tmp part of the name
    for filename in ${DEST_FOLDER}/*.dly-tmp; do
        mv "${filename}" "${filename%%-tmp}"
    done


'''

import os
from collections import namedtuple
import numpy as np
import random
import sys

EXAMPLE_COUNT_PER_STATION = 2000
MIN_COUNT_PER_STATION = 0

INCLUDED_ELEMENTS = ['TMAX', 'TMIN', 'PRCP']

SOURCE_FOLDER = sys.argv[1]  # .dly files, maybe trimmed by script in comments above
DESTINATION_FOLDER = sys.argv[2]  # output directory, should exist
DESTINATION_NPZ = os.path.join(
    DESTINATION_FOLDER,
    'gsn-{}-{}.npz'.format(
        EXAMPLE_COUNT_PER_STATION,
        '-'.join(INCLUDED_ELEMENTS)
    )
)
STATIONS_FILE = os.path.join(DESTINATION_FOLDER, 'stations')


RawStationData = namedtuple('RawStationData', [
    'station_id',
    'year',
    'month',
    'day',
    'element',
    'value',
])

DailyWeatherReport = namedtuple('DailyWeatherReport', [
    'station_id',
    'month',
    'day',
] + INCLUDED_ELEMENTS)


def count_lines_in_file(filename):
    total = 0
    with open(filename) as f:
        for line in f:
            total += 1

    return total


def parse_line_and_filter(line):
    '''Generator that gives valid RawStationDatas, skips missing and low-quality data.'''

    # magic numbers are from the readme!
    station_id = station_id=line[:11]
    year = line[11:15]
    month = line[15:17]
    element = line[17:21]

    # Don't bother processing the line if I don't need the data from it
    if element not in INCLUDED_ELEMENTS:
        return None

    # also from the readme, extract daily data for the 31 days!
    total_low_qual = 0
    for day in range(31):
        offset = 21 + (day * 8)
        value = line[offset:offset+5]
        mflag = line[offset+5]
        qflag = line[offset+6]
        sflag = line[offset+7]

        # throw out missing and low-quality data
        if value == '-9999':
            continue
        if qflag != ' ':
            # double check it's looking at the right column
            assert qflag in 'DGIKLMNORSTWXZ'
            continue

        yield RawStationData(
            station_id=station_id,
            year=int(year),
            month=int(month),
            day=day,
            element=element,
            value=int(value),
        )



def read_station_data(station_id_to_index, station_id, number_to_return):
    filename = os.path.join(SOURCE_FOLDER, '{}.dly'.format(station_id))

    # I'll use this dictionary to build up each DailyWeatherReport, representing a weather
    # report from a day/month/year.
    station_day = {}
    with open(filename) as f:
        for line in f:
            for entry in parse_line_and_filter(line):
                key = (entry.station_id, entry.year, entry.month, entry.day)
                if key not in station_day:
                    station_day[key] = DailyWeatherReport(
                        station_id=station_id_to_index[entry.station_id],
                        month=entry.month,
                        day=entry.day,
                        **{element: None for element in INCLUDED_ELEMENTS},  # initialize elements to None
                    )
                station_day[key] = station_day[key]._replace(**{entry.element: entry.value})

    # If no lines contained valid data, give up
    if not station_day:
        print('*{} no data!'.format(station_id))
        return

    # Quality control: count how many of each column are null or not.
    print(', '.join(
        '{} {:06f}'.format(
            element,
            sum(1 for d in station_day.values() if getattr(d, element) is not None)/len(station_day)
        )
        for element in INCLUDED_ELEMENTS
    ))

    # Filter out partial DailyWeatherReport, for example if the high temperature is low quality
    data = [
        list(weather)
        for weather in station_day.values()
        if None not in weather   # check that all values of DailyWeatherReport are non-null
    ]

    # Quality control, how many were filtered?
    print('{}: total valid {}, fraction valid {:06f}'.format(
        station_id,
        len(data),
        len(data)/len(station_day),
    ))

    if not data:
        print("*{}: No valid data for station!".format(station_id))
        return

    # convert to numpy array
    valid_days = np.vstack(data).astype(np.int32)

    if valid_days.shape[0] < MIN_COUNT_PER_STATION:
        print("*{}: Not enough valid data for station!".format(station_id))
        return
    if valid_days.shape[0] < number_to_return:
        # eh, probably doesn't matter, but shuffle them just in case.
        np.random.shuffle(valid_days)
        return valid_days
    else:
        # now choose a random subset
        indexes = np.random.choice(valid_days.shape[0], replace=False, size=number_to_return)
        return valid_days[indexes]


def make_dataset():
    print('writing to', DESTINATION_NPZ)

    # read in the stations, and write file containing the list of them
    stations = [
        name.split('.')[0]
        for name in os.listdir(SOURCE_FOLDER)
        if name.endswith('.dly')
    ]
    station_mapper = list(stations)

    # make a mapping from the station name to index
    station_name_to_index = {
        station_name: i
        for i, station_name in enumerate(station_mapper)
    }

    with open(STATIONS_FILE, 'w') as f:
        f.write('\n'.join(station_mapper))

    # do the big slow process of grabbing data from each station
    train_dataset_entries = (
        read_station_data(station_name_to_index, station, number_to_return=EXAMPLE_COUNT_PER_STATION)
        for station in station_mapper
    )

    data = np.vstack(
        entries
        for entries in train_dataset_entries
        if entries is not None
    ).astype(np.int16)

    np.savez(
        DESTINATION_NPZ,
        data=data,
    )


if __name__ == '__main__':
    make_dataset()
