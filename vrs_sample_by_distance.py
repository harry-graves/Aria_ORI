import os
import numpy as np
import csv

from projectaria_tools.core import data_provider, image
from projectaria_tools.core.stream_id import StreamId

vrsfile = "data/ET1.vrs"
provider = data_provider.create_vrs_data_provider(vrsfile)

def sample_poses_by_distance(input_csv_file, output_csv_file, sampling_distance=2.0):
    """
    Sample poses every 'sampling_distance' meters along the trajectory and save them to a new CSV file.
    
    input_csv_file: Path to the original .csv file containing the trajectory data.
    output_csv_file: Path to the output .csv file to store the sampled poses.
    sampling_distance: The distance interval in meters to sample poses.
    """
    with open(input_csv_file, 'r') as infile:
        csv_reader = csv.reader(infile)
        headers = next(csv_reader)  # Read header row

        sampled_rows = [headers]  # Start with the header for the output file
        prev_position = None

        for row in csv_reader:
            # Parse relevant data
            timestamp = float(row[1])  # tracking_timestamp_us (or any other time column you prefer)
            p_x, p_y, p_z = map(float, [row[3], row[4], row[5]])  # tx_world_device, ty_world_device, tz_world_device

            current_position = np.array([p_x, p_y, p_z])

            # Check if this is the first position or meets the distance requirement
            if prev_position is None:
                sampled_rows.append(row)
                prev_position = current_position
            else:
                distance = np.linalg.norm(current_position - prev_position)

                if distance >= sampling_distance:
                    sampled_rows.append(row)
                    prev_position = current_position

    # Write sampled poses to the new CSV file
    with open(output_csv_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerows(sampled_rows)
    
    print(f"Sampled poses have been saved to {output_csv_file}")

input_csv_file = 'data/trajectories/closed_loop_trajectory.csv'
output_csv_file = 'data/trajectories/sampled_poses.csv'
sample_poses_by_distance(input_csv_file, output_csv_file, sampling_distance=1.0)
