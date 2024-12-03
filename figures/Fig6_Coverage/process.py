import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/sparg")

import sparg
import tskit
import math
import numpy as np
import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt


# set seed
np.random.seed(10)


# load and filter the tree sequence
cutoff = 2000
ts = tskit.load("../../simulations/UniformStartTwoDims/outputs/original/S025_I1_R2_W100_D2/rep5_S025_I1_R2_W100_D2.trees")

locations = ts.tables.individuals.location
x = locations[::3]
y = locations[1::3]

# if using center sampling
#central_individuals = np.where((x - 150)**2 + (y - 150)**2 < 25**2)[0]
#appropriate_samples = np.where((ts.tables.nodes.flags == 1) & np.isin(ts.tables.nodes.individual, central_individuals))[0]
#samples = list(np.random.choice(appropriate_samples, 500, replace=False))

samples = list(np.random.choice(ts.samples(), 500, replace=False))
ts_sim, map_sim = ts.simplify(samples=samples, map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
ts_final, maps_final = sparg.simplify_with_recombination(ts=ts_sim, flag_recomb=True)
ts_chopped = sparg.chop_arg(ts=ts_final, time=cutoff)


# convert tree sequence to a SpatialARG
spatial_arg = sparg.SpatialARG(ts=ts_chopped, verbose=True)


# reset the dispersal rate matrix to the theoretical value
#spatial_arg.dispersal_rate_matrix = np.array([[0.25*0.25+0.5,0],[0,0.25*0.25+0.5]])


# create a dataframe of random ancestors in the ARG that we will estimate locations for
random_ancestors = sparg.generate_random_ancestors_dataframe(
    ts=ts_sim,
    number_of_ancestors=1000,
    cutoff=cutoff,
    include_locations=True,
    seed=10
)

# estimate locations using different windows and methods
random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_midpoint(
    df=random_ancestors,
    spatial_arg=spatial_arg,
    simplify=True
)
random_ancestors["midpoint_error_0"] = random_ancestors["true_location_0"] - random_ancestors["midpoint_estimated_location_0"]
random_ancestors["midpoint_abs_error_0"] = abs(random_ancestors["midpoint_error_0"])
random_ancestors["midpoint_error_1"] = random_ancestors["true_location_1"] - random_ancestors["midpoint_estimated_location_1"]
random_ancestors["midpoint_abs_error_1"] = abs(random_ancestors["midpoint_error_1"])
random_ancestors.to_csv("random_ancestors.csv")
print("Midpoint - Complete")

random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_arg(
    df=random_ancestors,
    spatial_arg=spatial_arg
)
random_ancestors["arg_error_0"] = random_ancestors["true_location_0"] - random_ancestors["arg_estimated_location_0"]
random_ancestors["arg_abs_error_0"] = abs(random_ancestors["arg_error_0"])
random_ancestors["arg_error_1"] = random_ancestors["true_location_1"] - random_ancestors["arg_estimated_location_1"]
random_ancestors["arg_abs_error_1"] = abs(random_ancestors["arg_error_1"])
random_ancestors.to_csv("random_ancestors.csv")
print("Full Chromosome - Complete")

random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(
    df=random_ancestors,
    spatial_arg=spatial_arg,
    window_size=0,
    use_theoretical_dispersal=False
)
random_ancestors["window_0_error_0"] = random_ancestors["true_location_0"] - random_ancestors["window_0_estimated_location_0"]
random_ancestors["window_0_abs_error_0"] = abs(random_ancestors["window_0_error_0"])
random_ancestors["window_0_error_1"] = random_ancestors["true_location_1"] - random_ancestors["window_0_estimated_location_1"]
random_ancestors["window_0_abs_error_1"] = abs(random_ancestors["window_0_error_1"])
random_ancestors.to_csv("random_ancestors.csv")
print("W0 - Complete")

random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(
    df=random_ancestors,
    spatial_arg=spatial_arg,
    window_size=100,
    use_theoretical_dispersal=False
)
random_ancestors["window_100_error_0"] = random_ancestors["true_location_0"] - random_ancestors["window_100_estimated_location_0"]
random_ancestors["window_100_abs_error_0"] = abs(random_ancestors["window_100_error_0"])
random_ancestors["window_100_error_1"] = random_ancestors["true_location_1"] - random_ancestors["window_100_estimated_location_1"]
random_ancestors["window_100_abs_error_1"] = abs(random_ancestors["window_100_error_1"])
random_ancestors.to_csv("random_ancestors.csv")
print("W100 - Complete")

exit()

random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(
    df=random_ancestors,
    spatial_arg=spatial_arg,
    window_size=200,
    use_theoretical_dispersal=True
)
random_ancestors["window_200_error_0"] = random_ancestors["true_location_0"] - random_ancestors["window_200_estimated_location_0"]
random_ancestors["window_200_abs_error_0"] = abs(random_ancestors["window_200_error_0"])
random_ancestors.to_csv("random_ancestors.csv")
print("W200 - Complete")

random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(
    df=random_ancestors,
    spatial_arg=spatial_arg,
    window_size=300,
    use_theoretical_dispersal=True
)
random_ancestors["window_300_error_0"] = random_ancestors["true_location_0"] - random_ancestors["window_300_estimated_location_0"]
random_ancestors["window_300_abs_error_0"] = abs(random_ancestors["window_300_error_0"])
random_ancestors.to_csv("random_ancestors.csv")
print("W300 - Complete")

random_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(
    df=random_ancestors,
    spatial_arg=spatial_arg,
    window_size=500,
    use_theoretical_dispersal=True
)
random_ancestors["window_500_error_0"] = random_ancestors["true_location_0"] - random_ancestors["window_500_estimated_location_0"]
random_ancestors["window_500_abs_error_0"] = abs(random_ancestors["window_500_error_0"])
random_ancestors.to_csv("random_ancestors.csv")
print("W500 - Complete")