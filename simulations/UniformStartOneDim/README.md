# Outputs

There are two folders in the outputs. "original" includes the tree sequence outputs from the SLiM simulation. These are often quite large as they track many individuals and have nodes at every generation. "simplified" includes corresponding ARGs are a subset of the original ARGs, with 500 samples, chopped at 2000 generations in the past, and removal of all nodes that do not affect the graph topology.

Within "original" and "simplified", there are folders which contain separate runs of the SLiM simulation under the specified parameters. For example, ".trees" files in the folder "S025_I1_R2_W100_D1" have:

- S025 : SIGMA_disp is set to 0.25
- I1 : SIGMA_int is set to 1
- R2 : R is set to 2
- W100 : W is set to 100
- D1 : One dimensional simulation

"simplified" ARGs also include an "N" parameter which is the random seed used when subsetting the samples.