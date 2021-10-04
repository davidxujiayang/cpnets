import sys
sys.path.append('..')

# xport LD_LIBRARY_PATH=/usr/local/tecplot/360ex_2018r2/bin/sys-util/:/usr/local/tecplot/360ex_2018r2/bin/:$LD_LIBRARY_PATH
from tecplotio import TecplotIO
from tecplot.constant import ValueLocation
import tecplot
import numpy as np

gridfile = 'data/grid.dat'
npyfile = 'data/cpgnet_pred.npy'

def get_outputname(i):
    return f'data/step_{i}.dat'

data = np.load(npyfile)
varnames = ['Static_Pressure', 'U', 'V', 'Temperature', 'CH4_mf', "O2_mf", "H2O_mf", "CO2_mf"]

dataset = tecplot.data.load_tecplot(gridfile, read_data_option=2)
for varname in varnames:
    dataset.add_variable(varname, locations=ValueLocation.CellCentered)

zone = dataset.zone(0)
var_save_list = ['x', 'y'] + varnames

for step in range(0, 41, 2):
    zone.solution_time = step
    for i, varname in enumerate(varnames):
        zone.values(varname)[:] = data[step,:,i]
    
    variables_to_save = [dataset.variable(V)
                        for V in var_save_list]
    tecplot.data.save_tecplot_plt(get_outputname(step), dataset=dataset,
                                    variables=variables_to_save,
                                    zones=zone)