import os
import glob
import json
from xml.etree import ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def process_xml(folder_name: str):
    path_pattern = os.path.join(folder_name, "tripinfo_*.xml")

    # Get the XML file paths, sort them, and select the last 10 files
    xml_files = sorted(glob.glob(path_pattern))

    avg_waiting_time = []

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
        total_waiting_time = 0
        trip_count = 0

        for tripinfo in root.findall('tripinfo'):
            trip_id = tripinfo.attrib['id']
            arrival = float(tripinfo.attrib['arrival'])
            waiting_time = float(tripinfo.attrib['timeLoss'])
            depart_delay = float(tripinfo.attrib['departDelay'])
            trip_time = float(tripinfo.attrib['duration'])
            trip_count += 1
            total_waiting_time += waiting_time

        avg_waiting_time.append(total_waiting_time / trip_count)

    return avg_waiting_time

if __name__ == '__main__':

    # Specify the base names of the folders for the RL and model-based control algorithms
    rl_base_folders = ["IDQN-training/IDQN-tr{}-ingolstadt7x-7-drq_norm-wait_norm",
                       "MPLight-training/MPLight-tr{}-ingolstadt7x-7-mplight-pressure"]
    mb_folders = ["MAXWAVE-tr5-ingolstadt7x-7-wave-pressure", "MAXPRESSURE-tr5-ingolstadt7x-7-mplight-pressure"]

    # Process the XML files in each folder and calculate the average waiting times
    rl_waiting_times = [[process_xml(base_folder.format(i + 1)) for i in range(5)] for base_folder in rl_base_folders]
    mb_waiting_times = [np.mean(process_xml(folder)) for folder in mb_folders]

    # For the RL algorithms, calculate the average and standard deviation across the 5 training trials for each episode
    rl_waiting_times_avg = [np.mean(times, axis=0) for times in rl_waiting_times]
    rl_waiting_times_std = [np.std(times, axis=0) for times in rl_waiting_times]

    # Plot the results for the RL algorithms
    rl_algorithms = ['IDQN', 'MPLight']
    for i, (avg_times, std_times) in enumerate(zip(rl_waiting_times_avg, rl_waiting_times_std)):
        episodes = range(1, len(avg_times) + 1)
        plt.plot(episodes, avg_times, label=rl_algorithms[i])
        plt.fill_between(episodes, avg_times - std_times, avg_times + std_times, alpha=0.2)

    # Plot the results for the model-based control algorithms
    mb_algorithms = ['LQF', 'MaxP']
    for i, avg_time in enumerate(mb_waiting_times):
        plt.axhline(y=avg_time, color='C{}'.format(i + len(rl_base_folders)), linestyle='--', label=mb_algorithms[i])

    plt.xlabel('Episode')
    plt.ylabel('Average Waiting Time')
    plt.legend()
    plt.grid()
    plt.show()

