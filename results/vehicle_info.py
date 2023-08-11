import os
import glob
import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def process_json(folder_name: str):
    path_pattern = os.path.join(folder_name, "vehicles_*.json")

    # Get the JSON file paths and sort them
    json_files = sorted(glob.glob(path_pattern))

    total_wait_dp = {'cluster_1757124350_1757124352': [], 'gneJ143': [], 'gneJ207': [],
 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': [],
 '32564122': [], 'gneJ260': [], 'gneJ210': []}

    vehicle_wait = {'cluster_1757124350_1757124352': {}, 'gneJ143': {}, 'gneJ207': {},
 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': {},
 '32564122': {}, 'gneJ260': {}, 'gneJ210': {}}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # check if vehicle exists in next time step, if not, add to total_wait_times
        for i in data:
            current_vehicles = i["vehicles"]
            for intersection in current_vehicles:
                intersection_vehicles = current_vehicles[intersection]
                if intersection_vehicles:
                    for id, params in intersection_vehicles.items():
                        vehicle_wait[intersection][id] = params["wait"]+0.1
    for intersection in vehicle_wait.keys():
        total_wait_dp[intersection] = list(vehicle_wait[intersection].values())
    return total_wait_dp


def plot_kde(data_dicts, labels, colors, title='Distribution of Pace', xlabel='Seconds'):
    # Ensure that the number of labels match with the number of data_dicts
    assert len(data_dicts) == len(labels), "Number of labels should match with number of data dicts."

    # Check if the colors list is as long as the labels list. If not, repeat it
    if len(colors) < len(labels):
        colors = colors * len(labels)

    # Create a grid for subplots
    fig, axs = plt.subplots(4, 2, figsize=(8, 14))
    axs = axs.flatten()  # flatten the axes array for easy iteration

    intersections = data_dicts[0].keys()

    # Iterate over each intersection and plot its KDE
    for ax, intersection, idx in zip(axs, intersections, range(1, 8)):
        for i, (data_dict, label) in enumerate(zip(data_dicts, labels)):
            data_points = data_dict[intersection]
            log_data = np.log10(np.array(data_points) + 1e-10)

            # Plot the KDE
            sns.kdeplot(log_data, alpha=0.5, color=colors[i], label=label, ax=ax)

        ax.set_yscale('log')
        ax.set_xlim([1, 10])
        ax.set_ylim([0.001, 1])
        ax.set_title(f'Intersection {idx}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')

    # Delete the remaining subplots if there are less than 8 intersections
    for i in range(len(intersections), len(axs)):
        fig.delaxes(axs[i])

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_box(data_lists, labels, title='Box Plot of Pace', xlabel='Algorithm', ylabel='second'):
    # Ensure that the number of labels match with the number of data_lists[0] (number of algorithms)
    assert len(data_lists[0]) == len(labels), "Number of labels should match with number of data dicts."

    # Number of demand levels
    num_demands = len(data_lists)

    # Number of intersections
    num_intersections = len(data_lists[0][0])

    # Create a grid for subplots
    fig, axs = plt.subplots(num_intersections, num_demands, figsize=(15, 3 * num_intersections))

    # Iterate over each intersection and demand level
    for i, intersection in enumerate(data_lists[0][0].keys()):
        for j, demand_data in enumerate(data_lists):
            ax = axs[i, j]

            # Prepare data for seaborn boxplot
            data = []
            group = []
            for k, algorithm_data in enumerate(demand_data):
                data_points = algorithm_data[intersection]
                log_data = np.log10(np.array(data_points) + 1e-10)

                data.extend(data_points)
                group.extend([labels[k]] * len(data_points))

            # Plot the Box plot
            sns.boxplot(x=group, y=data, ax=ax)

            # Compute the IQR and set the y-limits
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            ax.set_ylim([Q1 - 2 * IQR, min(10, Q3 + 2 * IQR)])

            if j == 0:
                ax.set_ylabel(ylabel)

            if i == 0:
                ax.set_title(['Low Demand', 'Medium Demand', 'High Demand'][j])

    plt.tight_layout()
    plt.show()


def plot_violin(data_lists, labels, title='Violin Plot of Pace', xlabel='Algorithm', ylabel='second'):
    # Ensure that the number of labels match with the number of data_lists[0] (number of algorithms)
    assert len(data_lists[0]) == len(labels), "Number of labels should match with number of data dicts."
    plt.rcParams.update({'font.size': 14})
    # Number of demand levels
    num_demands = len(data_lists)

    # Number of intersections
    num_intersections = len(data_lists[0][0])

    # Create a grid for subplots
    fig, axs = plt.subplots(num_intersections, num_demands, figsize=(11, 2.5 * num_intersections))

    # Iterate over each intersection and demand level
    for i, intersection in enumerate(data_lists[0][0].keys()):
        for j, demand_data in enumerate(data_lists):
            ax = axs[i, j]

            # Prepare data for seaborn violinplot
            data = []
            group = []
            for k, algorithm_data in enumerate(demand_data):
                data_points = algorithm_data[intersection]
                data.extend(data_points)
                group.extend([labels[k]] * len(data_points))

            # Plot the Violin plot
            sns.violinplot(x=group, y=data, ax=ax)

            # Set the upper y-limit based on Q3 + 2*IQR
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            ax.set_ylim([-5, max(10, Q3 + 2 * IQR)])

            if j == 0:
                ax.set_ylabel(ylabel, fontsize=14)

            if i == 0:
                ax.set_title(['Low Demand', 'Medium Demand', 'High Demand'][j])

        fig.text(1.0, (num_intersections - i - 0.5) / num_intersections, f"Intersection {i + 1}", ha='right',
                 va='center', rotation=90, fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demands = ['low', 'mid', 'hig']
    alg_names = ['IDQN', 'MPLight', 'MaxP', 'LQF']
    colors = ['blue', 'orange', 'green', 'red']
    title = 'Pace - High Demand'
    datapoints = []
    for demand in demands:
        folder_names = [f"IDQN-tr5-ingolstadt7{demand}-7-drq_norm-wait_norm",
                        f"MPLight-tr3-ingolstadt7{demand}-7-mplight-pressure",
                        f"MAXPRESSURE-tr3-ingolstadt7{demand}-7-mplight-pressure",
                        f"MAXWAVE-tr3-ingolstadt7{demand}-7-wave-pressure"]
        datapoints.append([process_json(folder) for folder in folder_names])

    plot_violin(datapoints, alg_names)
