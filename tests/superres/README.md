The original DNS data can be found at https://turbulence.oden.utexas.edu/, thanks to the work of 
Prof. J. Jimenez, Prof. R. D. Moser, Dr. J.-C. del Alamo and Dr. P. S. Zandonade.

1. The DNS snapshots are provided under 'data/snap*.mat', where * is replaced with different normalized wall normal heights.
2. Run dg_datagen.m to project to dg space
3. Run run_mlp.py and run_cpmlp.py to train and test the models.
4. Run recon_from_dg.m to reconstruct prediction results
5. Run recon_to_npy.py to convert matlab data to python friendly files
6. Run pred_combined_pred.py for a visualization