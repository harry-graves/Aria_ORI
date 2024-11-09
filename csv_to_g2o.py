import csv

# Function to convert timestamp in microseconds to sec and nanosec
def timestamp_to_sec_nsec(timestamp_us):
    timestamp_sec = int(timestamp_us // 1e6)
    timestamp_nsec = int((timestamp_us % 1e6) * 1e3)
    return timestamp_sec, timestamp_nsec

# Read closed loop trajectory data and convert it to the g2o format
input_file = 'data/trajectories/closed_loop_trajectory.csv'
output_file = 'data/trajectories/slam_pose_graph.g2o'

# Lists to store vertices and edges
vertices = []
edges = []

with open(input_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Initialize variables to store the previous pose for EDGE_SE3:QUAT entries
    prev_id, prev_tx, prev_ty, prev_tz = None, None, None, None
    prev_qx, prev_qy, prev_qz, prev_qw = None, None, None, None

    # Iterate over rows to create VERTEX and EDGE entries
    for idx, row in enumerate(reader):
        # Extract pose position and orientation data
        tracking_timestamp_us = int(row['tracking_timestamp_us'])
        tx, ty, tz = float(row['tx_world_device']), float(row['ty_world_device']), float(row['tz_world_device'])
        qx, qy, qz, qw = float(row['qx_world_device']), float(row['qy_world_device']), float(row['qz_world_device']), float(row['qw_world_device'])
        
        # Convert timestamp for VERTEX_SE3:QUAT_TIME
        sec, nsec = timestamp_to_sec_nsec(tracking_timestamp_us)
        
        # Add VERTEX_SE3:QUAT_TIME entry to the list
        vertices.append(f"VERTEX_SE3:QUAT_TIME {idx} {tx} {ty} {tz} {qx} {qy} {qz} {qw} {sec} {nsec}")
        
        # If we have a previous entry, add EDGE_SE3:QUAT constraint
        if prev_id is not None:
            # Example transformation values between poses (can be set or computed)
            # These can be refined based on actual system specifics
            edges.append(f"EDGE_SE3:QUAT {prev_id} {idx} {tx-prev_tx} {ty-prev_ty} {tz-prev_tz} "
                         f"{qx-prev_qx} {qy-prev_qy} {qz-prev_qz} {qw-prev_qw} "
                         "1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 1.0")
        
        # Update previous pose data for next EDGE
        prev_id, prev_tx, prev_ty, prev_tz = idx, tx, ty, tz
        prev_qx, prev_qy, prev_qz, prev_qw = qx, qy, qz, qw

# Write vertices and edges to output file
with open(output_file, 'w') as g2ofile:
    # Write initial settings (LLA reference, Platform ID)
    g2ofile.write("# GNSS_LLA_REF latitude longitude altitude\n")
    g2ofile.write("# Replace with actual LLA values if available\n")
    g2ofile.write("GNSS_LLA_REF 0 0 0\n\n")
    g2ofile.write("PLATFORM_ID 0\n\n")
    
    # Write all vertices first
    for vertex in vertices:
        g2ofile.write(vertex + "\n")
    
    # Write all edges after vertices
    for edge in edges:
        g2ofile.write(edge + "\n")

print(f"File '{output_file}' generated successfully.")
