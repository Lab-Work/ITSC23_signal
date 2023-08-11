

import os
import json
import csv
import numpy as np
import ast

input_folder = "IDQN-tr3-ingolstadt7x-7-drq_norm-wait_norm"
reward_file = "IDQN_reward_ingolstadt7x.json"
queue_file = "IDQN_queue_ingolstadt7x.json"


# Initialize empty dictionaries for rewards and queue lengths
rewards = {'cluster_1757124350_1757124352': [], 'gneJ143': [], 'gneJ207': [],
 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': [],
 '32564122': [], 'gneJ260': [], 'gneJ210': []}

queue_lengths = {'cluster_1757124350_1757124352': [], 'gneJ143': [], 'gneJ207': [],
 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': [],
 '32564122': [], 'gneJ260': [], 'gneJ210': []}

for episode in range(1, 101):
    input_file = os.path.join(input_folder, f"metrics_{episode}.csv")

    # Initialize variables to calculate the episode averages
    reward_sums = {'cluster_1757124350_1757124352': 0.0, 'gneJ143': 0.0, 'gneJ207': 0.0,
 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': 0.0,
 '32564122': 0.0, 'gneJ260': 0.0, 'gneJ210': 0.0}

    queue_length_sums = {'cluster_1757124350_1757124352': 0.0, 'gneJ143': 0.0, 'gneJ207': 0.0,
 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': 0.0,
 '32564122': 0.0, 'gneJ260': 0.0, 'gneJ210': 0.0}
    with open(input_file) as csvfile:
        # reader = csv.reader(csvfile)
        timesteps = 0
        for row in csvfile:
            row = row.split('}')[:-1]
            reward, max_queue, queue_len = row[0], row[1], row[2]
            # ["20.0, {'J3': -1, 'J4': -2, 'J5': -1, 'J7': -1", ", {'J3': 1, 'J4': 1, 'J5': 1, 'J7': 1", ", {'J3': 1, 'J4': 2, 'J5': 1, 'J7': 1"]

            reward = reward.split("{")[-1]
            max_queue = max_queue.split('{')[-1]
            queue_len = queue_len.split("{")[-1]
            converted_reward = ast.literal_eval("{"+reward+"}")
            converted_max_queue = ast.literal_eval("{"+max_queue+"}")
            converted_queue_len = ast.literal_eval("{"+queue_len+"}")

            for intersection, value in converted_reward.items():
                reward_sums[intersection] += value
            for intersection, value in converted_queue_len.items():
                queue_length_sums[intersection] += value
            timesteps += 1

    # Calculate averages and append to the corresponding lists
    for intersection in reward_sums:
        rewards[intersection].append(reward_sums[intersection] / timesteps)
        queue_lengths[intersection].append(queue_length_sums[intersection] / timesteps)
print("Conversion completed! Saving outputs...")
# Save the results as JSON files
with open(reward_file, 'w') as outfile:
    json.dump(rewards, outfile)

with open(queue_file, 'w') as outfile:
    json.dump(queue_lengths, outfile)
