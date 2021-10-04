The dataset can be found at https://afcoe.engin.umich.edu/benchmark-data, thanks to the work of Dr. Cheng Huang, et al. 

1. Sample the solution snapshots at desired interval (5 in our test) and stack into a numpy array with shape (nstep x npoint x nvariable). To do this, you can use the helper files provided along with the data. We also provide an example helper file tecplotio.py that uses pytecplot for the purpose. To use the rest of our code, the snapshot stack should be stored under 'data/deepblue_snapshots_int5_len6000.npy', where int5 indicates the interval and len6000 indicates the total sampled steps. 
2. Run grid_to_graph.py to build graph from grid.dat (provided in the dataset).
3. Run run_cpgnet.py and run_gnet.py to train and test the models.
4. npy_to_plt.py serves as an example tool to convert python prediction into tecplot files for visualization.