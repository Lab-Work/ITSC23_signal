# SUMO-TSC-evaluation

This repo includes the source data & code for our paper, "Evaluation of Traffic Signal Control at Varying Demand Levels: A Comparative Study", in IEEE ITSC 2023.

The code structure is based on [RESCO Benchmark](https://github.com/Pi-Star-Lab/RESCO). We have modified/added functionalities for our paper use.

## Structure

- `pfrl` is a local package of [pfrl](https://github.com/pfnet/pfrl) modified for model testing purposes.
- `resco_benchmark` is the modified SUMO-based traffic signal control package with various useful built-in functionalities. We make modifications as follows:
  - agent_tf2.0: we convert all tensorflow 1.x uses to a tf2.x-compatible version.
  - Scenario: We modified the original Ingolstadt scenario to make it work better with TSC algorithms. Besides, we fixed some map inconsistencies in `signal_config.py`.
  - Demands: we created 3 static and 1 time-varying demand files for our evaluation. They are named as `ingolstadt7low`, `ingolstadt7mid`, `ingolstadt7hig` (static) and `ingolstadt7x` (dynamic). 
  - Output: Vehicle data is retrieved as output from SUMO config. We enabled the retrieval of vehicle (trip) data and unfinished trips.
- `results` includes all training and testing results from our experiments, in which `pace_plotting.py`, `training_plotting.py`, `vehicle_info.py` are three visualization scripts. `xml_processing.py` and `csv_processing_ing7.py` are postprocessing scripts for SUMO output data.

## Usage

For algorithm training and testing, run `resco_benchmark/main.py` with corresponding parameters. For output analysis and visualizations, use the scripts in `results/`.

*Results are kept in Vanderbilt University Institutional Repository ([link](http://hdl.handle.net/1803/18314)). Unzip "data.zip" in 'results/' for processing.

## Contact

- Author: Zhiyao Zhang, Marcos Quinones-Grueiro, William Barbour, Yuhang Zhang, Gautam Biswas, and Daniel Work
- Affiliation: Institute for Software Integrated Systems, Vanderbilt University
- First-author email: zhiyao.zhang@vanderbilt.edu
