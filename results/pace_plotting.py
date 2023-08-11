import os
import glob
import json
from xml.etree import ElementTree as ET
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def process_xml(folder_name:str):
    path_pattern = os.path.join(folder_name, "tripinfo_*.xml")

    # Get the XML file paths, sort them, and select the last 10 files
    xml_files = sorted(glob.glob(path_pattern))

    total_pace = list()

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


        for tripinfo in root.findall('tripinfo'):
            trip_id = tripinfo.attrib['id']
            arrival = float(tripinfo.attrib['arrival'])
            waiting_time = float(tripinfo.attrib['timeLoss'])
            depart_delay = float(tripinfo.attrib['departDelay'])
            trip_time = float(tripinfo.attrib['duration'])

            if arrival != -1.00:
                pace = trip_time / float(tripinfo.attrib['routeLength'])
                total_pace.append(pace)

            elif trip_id in unfinished_dict:  # Check if the trip id is in the dictionary keys
                distance = unfinished_dict[trip_id]
                if distance > 1:
                    pace = trip_time / (distance)
                    # print(f"trip {trip_id}, pace {pace}, trip_time {trip_time}, distance {distance}")
                    total_pace.append(pace)  # Update the total pace with the unfinished trip's pace
            else:
                print(f"Warning: Unfinished trip {trip_id} not found in the JSON file.")

    assert total_pace, "Warning: no pace datapoint found!"

    return total_pace


def plot_kde(data_points, labels, colors, title='Distribution of Pace', xlabel='Pace'):

    # Plot the KDE
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, data in enumerate(data_points):
        log_data = np.log10(np.array(data) + 1e-10)
        sns.kdeplot(log_data, alpha=0.5, color=colors[i], label=labels[i])
    ax.set_xscale('log')

    plt.title(title)
    plt.xlim([0.01, 1000])
    plt.ylim([0, 0.5])
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000])
    plt.legend()
    plt.show()


def plot_violin(data_points, labels, title='Box Plot of Pace', xlabel='Algorithm', ylabel='log(second/meter)'):
    # Increase default font size
    plt.rcParams.update({'font.size': 14})

    # Convert data to logarithmic scale to avoid negative values
    log_data_points = [np.log10(np.array(data) + 1e-10) for data in data_points]

    # Prepare data for seaborn boxplot
    data = []
    group = []
    avg_values = []
    std = []
    outlier_counts = []
    for i, (data_pt, log_data, label) in enumerate(zip(data_points, log_data_points, labels)):
        data.extend(log_data)
        group.extend([label]*len(log_data))
        avg_values.append(np.mean(data_pt))
        std.append(np.std(data_pt))

        Q1, Q3 = np.percentile(log_data, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = [x for x in log_data if x < lower_bound or x > upper_bound]
        outlier_counts.append(len(outliers))

    # Plot the Box plot
    fig, ax = plt.subplots(figsize=(4.8, 4))  # Adjust the figure size
    sns.violinplot(x=group, y=data, ax=ax)
    ax.set_ylim([-1.5, None])

    ax.set_title(title)
    # ax.set_ylabel(ylabel)

    # for i, (avg_value, outlier_count) in enumerate(zip(avg_values, outlier_counts)):
    #     ax.text(i, -1.3, f"mean:\n{10 ** avg_value:.4f}", ha='center', va='top',
    #             fontsize=12, color='black')
    plt.tight_layout()
    plt.show()
    print(avg_values, std)

if __name__ == '__main__':
    demand = 'low'
    folder_names = [f"IDQN-tr5-ingolstadt7{demand}-7-drq_norm-wait_norm",
                    f"MPLight-tr3-ingolstadt7{demand}-7-mplight-pressure",
                    f"MAXPRESSURE-tr3-ingolstadt7{demand}-7-mplight-pressure",
                    f"MAXWAVE-tr3-ingolstadt7{demand}-7-wave-pressure"]
    alg_names = ['IDQN', 'MPLight', 'MaxP', 'LQF']
    colors = ['yellow', 'orange', 'green', 'pink']
    title = 'High Demand'
    pace_dp = [process_xml(folder_name) for folder_name in folder_names]
    # plot_kde(pace_dp, alg_names, colors, title)
    plot_violin(pace_dp, alg_names, title)