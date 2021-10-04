import os
import numpy as np
import argparse
import tecplot
from tecplot.constant import ValueLocation


def get_filename(directory, fidx):
    filename = os.path.join(directory, "test_file_ncons_%i.dat" % fidx)
    return filename
    
class TecplotIO(object):
    def __init__(self):
        pass

    def read_data(self, filename, cell2node, varnames):
        print(filename)
        dataset = tecplot.data.load_tecplot(filename, read_data_option=2)
        zone = dataset.zone(0)
        num_variables = zone.num_variables
        num_points = zone.num_points
        num_elements = zone.num_elements
        if cell2node == 0 or cell2node == 3:
            ncv = num_elements
        else:
            ncv = num_points
        time = zone.solution_time
        variables = list(zone.dataset.variables())
        data = self.read_varnames_to_dict(dataset, varnames,
                                          ncv, cell2node)

        output = [data]
        return output

    def read_varnames_to_dict(self, dataset, varnames, ncv, cell2node):
        # cell2node = 0  for cell data as-is, 1 for convert cell to node, 2 for node data as-is, 3 for convert node to cell
        zone = dataset.zone(0)
        variables = list(zone.dataset.variables())
        nvar = len(varnames)
        data = np.zeros([ncv, nvar], dtype=np.float64)
        for v in variables:
            if v.name in varnames:
                idx = varnames.index(v.name)
                if cell2node == 1:
                    tecplot.data.operate.execute_equation(equation='{'+v.name+'_node}={'+v.name+'}',
                                                          value_location=ValueLocation.Nodal)
                    data[:, idx] = zone.values(v.name+'_node').as_numpy_array().astype(
                        np.float64)
                elif cell2node == 3:
                    tecplot.data.operate.execute_equation(equation='{'+v.name+'_cell}={'+v.name+'}',
                                                          value_location=ValueLocation.CellCentered)
                    data[:, idx] = zone.values(v.name+'_cell').as_numpy_array().astype(
                        np.float64)                    
                else:
                    data[:, idx] = zone.values(v.name).as_numpy_array().astype(
                        np.float64)
        return data

    def query_ncv(self, filename, cell2node):
        dataset = tecplot.data.load_tecplot(filename, read_data_option=2)
        zone = dataset.zone(0)
        num_variables = zone.num_variables
        num_points = zone.num_points
        num_elements = zone.num_elements
        if cell2node == 0 or cell2node == 3:
            ncv = num_elements
        else:
            ncv = num_points
        return ncv

    def read_plts(self,
                  varnames,
                  start=1,
                  end=100,
                  step=1,
                  directory=None,
                  cell2node=0):
        filename = get_filename(directory, start)
        print(filename)
        ncv = self.query_ncv(filename, cell2node)
        print(ncv)
        nvar = len(varnames)
        nt = len(range(start, end+step, step))
        data_array = np.zeros([nt, ncv, nvar], dtype=np.float64)

        for fidx, time in enumerate(range(start, end+step, step)):
            print(f'Loading plts {fidx}')
            output_data = self.read_data(
                get_filename(directory, time), cell2node)
            data_array[fidx, :, :] = output_data[0][:, :]

        return data_array