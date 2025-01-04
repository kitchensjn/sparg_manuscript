# Boundary Effects

This folder mirrors Figure 6 from the manuscript but does so with a simulation run in a larger area. Samples are pulled only from the center, thus reducing the influence of the reflecting boundary walls. This helps show that the observed patterns are not a consequence of boundary effects.

"process.py" runs sparg on a tree sequence output by SLiM. This generates the random_ancestors.csv file which is used in "S_BoundaryEffects.ipynb" to create accuracy and coverage plots. Subfigures of S_BoundaryEffects (along with additional supplemental figures) are stored within the "subfigures" folder.