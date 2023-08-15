import os
import glob
import json
from xml.etree import ElementTree as ET

folder_name = "IDQN-tr5-ingolstadt7hig-7-drq_norm-wait_norm"
path_pattern = os.path.join(folder_name, "tripinfo_*.xml")

# Get the XML file paths, sort them, and select the last 10 files
xml_files = sorted(glob.glob(path_pattern))

avg_trip_time = []
avg_waiting_time = []
avg_arrival_count = []
avg_time_loss = []
avg_pace = []

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # print("\n", xml_file)
    # Load the corresponding JSON file
    json_index = xml_file.split('_')[-1].split('.')[0]  # Extract the index from the xml filename
    # print(json_index)
    json_file = os.path.join(folder_name, f"unfinished_{json_index}.json")
    with open(json_file, 'r') as f:
        unfinished_dict = json.load(f)

    arrived_trips_count = 0
    total_trip_time = 0
    total_waiting_time = 0
    trip_count = 0
    total_timeloss = 0
    total_pace = 0

    for tripinfo in root.findall('tripinfo'):
        trip_id = tripinfo.attrib['id']
        arrival = float(tripinfo.attrib['arrival'])
        waiting_time = float(tripinfo.attrib['timeLoss'])
        depart_delay = float(tripinfo.attrib['departDelay'])
        trip_time = float(tripinfo.attrib['duration'])
        if trip_time < 1: trip_time += 1    # avoid infinity
        norm_timeloss = (waiting_time + depart_delay) / trip_time
        trip_count += 1

        total_timeloss += norm_timeloss
        if arrival != -1.00:
            pace = trip_time / float(tripinfo.attrib['routeLength'])
            total_pace += pace
            total_trip_time += trip_time
            total_waiting_time += waiting_time
            arrived_trips_count += 1
        elif trip_id in unfinished_dict:  # Check if the trip id is in the dictionary keys
            distance = unfinished_dict[trip_id]
            if distance > 1:
                pace = trip_time / (distance)
                # print(f"trip {trip_id}, pace {pace}, trip_time {trip_time}, distance {distance}")
                total_pace += pace  # Update the total pace with the unfinished trip's pace
        else:
            print(f"Warning: Unfinished trip {trip_id} not found in the JSON file.")

    avg_time_loss.append(total_timeloss / trip_count)
    avg_waiting_time.append(total_waiting_time / trip_count)
    avg_pace.append(total_pace / trip_count)
    avg_trip_time.append(total_trip_time / arrived_trips_count)
    avg_arrival_count.append(arrived_trips_count)

if avg_waiting_time:
    print(f"Average trip time: {avg_trip_time}")
    print(f"Average wait time: {avg_waiting_time}")
    print(f"Average time loss (normalized): {avg_time_loss}")
    print(f"Average pace: {avg_pace}")
    print(f"Number of arrived trips: {avg_arrival_count}")
else:
    print("No arrived trips found.")

