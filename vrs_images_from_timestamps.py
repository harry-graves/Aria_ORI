import os
import numpy as np
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
import csv
import matplotlib.pyplot as plt

def images_from_timestamps(vrs_file, csv_file, output_dir, camera_label):

    # Initialize provider
    print(f"Creating data provider from {vrs_file}")
    provider = data_provider.create_vrs_data_provider(vrs_file)
    if not provider:
        print("Invalid vrs data provider")

    # Initialize output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize data provider
    stream_id = provider.get_stream_id_from_label(camera_label)
    time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
    option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
    start_time = provider.get_first_time_ns(stream_id, time_domain)
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    pinhole = calibration.get_linear_camera_calibration(512, 512, 150, camera_label,calib.get_transform_device_camera())

    # Read in the sampled timestamps from the sampled .csv trajectory file
    timestamps = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header if there is one
        for row in csv_reader:
            timestamps.append(int(row[1]) * 1000)  # Index 1 for the second column

    # Iterate through each row of the sampled trajectory
    index = 1
    for timestamp in timestamps:
        image = provider.get_image_data_by_time_ns(stream_id, timestamp, time_domain, option)
        image_array = image[0].to_numpy_array()

        rectified_array = calibration.distort_by_calibration(image_array, pinhole, calib)
        rotated_array = np.rot90(rectified_array, k=3)

        plt.imsave(f'{output_dir}/image_{index}.png', rotated_array, cmap='gray')
        index += 1

# Set paths and parameters
csv_file = "data/trajectories/sampled_poses.csv"
vrs_file = "data/ET1.vrs"
output_dir = "data/sampled_images"
camera_label = "camera-rgb"

images_from_timestamps(vrs_file, csv_file, output_dir, camera_label)