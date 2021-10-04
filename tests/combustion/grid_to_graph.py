import sys
sys.path.append('../../src/')
import scipy.spatial as ss
import networkx as nx
import numpy as np
from misc import *
import tecplot
from tecplotio import TecplotIO
from numpy.linalg import norm


def compute_volume(points):
    hull = ss.ConvexHull(points)
    volume = hull.volume
    assert volume>0
    return volume


def load_tecplot_grid(fn, points_per_face=2):
    varnames = ['x', 'y']

    dataset = tecplot.data.load_tecplot(fn, read_data_option=2)
    zone = dataset.zone(0)
    nvar = zone.num_variables
    npt = zone.num_points
    nelem = zone.num_elements
    nodemap = zone.nodemap
    npe = nodemap.num_points_per_element

    t = TecplotIO()
    coords = t.read_varnames_to_dict(dataset, varnames, nelem, cell2node=3)
    print(f'coords shape: {coords.shape}')
    xmax = coords[:,0].max()
    xmin = coords[:,0].min()
    deltax = (xmax-xmin)/400
    xfuel0 = -0.032
    xfuel1 = -0.029

    pt_coords = t.read_varnames_to_dict(dataset, varnames, npt, cell2node=2)
    print(f'pt_coords shape: {pt_coords.shape}')

    faces = []
    volumes = []
    areas = []
    boundary_elems = []
    norm_vecs = []
    distances = []
    wall_edges = []

    for elemid in range(nelem):
        if elemid%500==0:
            print(elemid)
        nodes = nodemap[elemid]
        assert len(nodes)==4
        volumes.append(compute_volume(pt_coords[nodes, ]))
        # print('volume points: ', nodes)
        element_checked = [elemid]
        neighbor_elems = []

        shared_counts = {}
        for nodeid in nodes:
            shared_counts[nodeid] = shared_counts.get(nodeid, 0)
            nep = nodemap.num_elements(nodeid)
            for i in range(nep):
                next_elem = nodemap.element(nodeid, i)
                if next_elem not in element_checked:
                    element_checked.append(next_elem)
                    next_nodes = nodemap[next_elem]
                    shared_points = list(set(nodes).intersection(next_nodes))
                    if len(shared_points) == points_per_face:
                        neighbor_elems.append(next_elem)
                        shared_counts[shared_points[0]] = shared_counts.get(shared_points[0], 0) + 1
                        shared_counts[shared_points[1]] = shared_counts.get(shared_points[1], 0) + 1

        neighbor_elems = list(set(neighbor_elems))

        if len(neighbor_elems) < len(nodes):
            # More connected points than cells, this is a boundary point
            shared_counts = {k: v for k, v in sorted(shared_counts.items(), key=lambda item: item[1])}
            gap = (len(nodes) - len(neighbor_elems))
            # print(gap, shared_counts, shared_counts.values())
            assert gap < 3
            if gap == 1:
                assert list(shared_counts.values())==[1,1,2,2]
            else:
                assert list(shared_counts.values())==[0,1,1,2]
            xelem = coords[elemid,0]

            if xelem<xmin+deltax or xelem>xmax-deltax or (xelem > xfuel0 and xelem < xfuel1):
                vec = pt_coords[list(shared_counts)[0], ] - pt_coords[list(shared_counts)[1], ]
                vec = vec/norm(vec)
                if np.abs(vec[0]) > 0.9:
                    wall_edges.append([elemid, list(shared_counts)[0], list(shared_counts)[1]])
                else:
                    boundary_elems.append(elemid)
                if gap > 1:
                    vec = pt_coords[list(shared_counts)[0], ] - pt_coords[list(shared_counts)[2], ]
                    vec = vec/norm(vec)
                    if np.abs(vec[0]) > 0.9:
                        wall_edges.append([elemid, list(shared_counts)[0], list(shared_counts)[2]])
                    else:
                        boundary_elems.append(elemid)
            else:
                wall_edges.append([elemid, list(shared_counts)[0], list(shared_counts)[1]])
                if gap > 1:
                    wall_edges.append([elemid, list(shared_counts)[0], list(shared_counts)[2]])

        neighbor_elems = [
            next_elem for next_elem in neighbor_elems if next_elem > elemid]

        for next_elem in neighbor_elems:
            faces.append([elemid, next_elem])
            next_nodes = nodemap[next_elem]
            shared_points = list(set(nodes).intersection(next_nodes))
            
            vec = pt_coords[shared_points[0], ]-pt_coords[shared_points[1], ]
            area = norm(vec)
            norm_vec = np.array([vec[1], -vec[0]])/area
            elem_vec = coords[elemid] - coords[next_elem]
            d = np.dot(elem_vec, norm_vec)
            if d<0:
                norm_vec = -norm_vec
                d = -d
                
            areas.append(area)
            norm_vecs.append(norm_vec)
            distances.append(d)

    wall_edges_noslip = []
    wall_edges_symmetry = []

    for wall_edge in wall_edges:
        if coords[wall_edge[0], 1] > 0.005:
            wall_edges_noslip.append(wall_edge)
        else:
            wall_edges_symmetry.append(wall_edge)

    wall_edges = wall_edges_noslip + wall_edges_symmetry
    N_ie = len(faces)
    N_ns = len(wall_edges_noslip)
    N_sym = len(wall_edges_symmetry)
    print('N inner: ', N_ie, ', N no slip: ', N_ns, ', N symmetry: ', N_sym)

    for i, wall_edge in enumerate(wall_edges):
        faces.append([wall_edge[0], nelem+i])
        shared_points = wall_edge[1:]
        
        vec = pt_coords[shared_points[0], ]-pt_coords[shared_points[1], ]
        area = norm(vec)
        norm_vec = np.array([vec[1], -vec[0]])/area
        elem_vec = coords[wall_edge[0]] - 0.5*(pt_coords[shared_points[0], ] + pt_coords[shared_points[1], ])

        d = 2*np.dot(elem_vec, norm_vec)

        if d<0:
            norm_vec = -norm_vec
            d = -d
            
        areas.append(area)
        norm_vecs.append(norm_vec)
        distances.append(d)

    boundary_elems = list(set(boundary_elems))

    faces = np.asanyarray(faces)
    print(f'faces shape: {faces.shape}')
    areas = np.asanyarray(areas)
    print(f'areas shape: {areas.shape}')
    volumes = np.asanyarray(volumes)
    print(f'volumes shape: {faces.shape}')
    norm_vecs = np.asanyarray(norm_vecs)
    print(f'norm_vecs shape: {norm_vecs.shape}')


    return faces, coords, areas, volumes, boundary_elems, norm_vecs, distances


def build_graph_from_tecplot(fn, plot=False):
    # edge in grapn is normal to faces in CFD mesh
    edges, coords, areas, volumes, boundary_elems, norm_vecs, distances = load_tecplot_grid(
        fn)
    unq, count = np.unique(coords, axis=0, return_counts=True)
    print(coords.shape, max(count))
    ncell = coords.shape[0]
    nedge = edges.shape[0]
    print(f'ncell: {ncell}, nedge: {nedge}')

    # build the graph
    g = nx.MultiDiGraph()
    cells = np.arange(0, ncell, dtype=int)
    g.add_nodes_from(cells)
    for i in range(ncell):
        g.nodes[i]['coords'] = coords[i, :2]
        g.nodes[i]['volume'] = volumes[i]
        g.nodes[i]['ghost'] = 0

        if i in boundary_elems:
            g.nodes[i]['is_boundary'] = 1
        else:
            g.nodes[i]['is_boundary'] = 0

    for i in range(nedge):
        g.add_edge(edges[i, 0], edges[i, 1], area=areas[i], d=distances[i], vec=norm_vecs[i,], idx=i)
        if edges[i, 1]>=ncell:
            g.add_node(edges[i, 1], coords=coords[edges[i, 0], :2]-distances[i]*norm_vecs[i,], volume=volumes[edges[i, 0]], is_boundary = 0)
            if g.nodes[edges[i, 1]]['coords'][1] > 0.005:
                g.nodes[edges[i, 1]]['ghost'] = 1
            else:
                g.nodes[edges[i, 1]]['ghost'] = 2

    return g

if __name__ == '__main__':
    g = build_graph_from_tecplot(fn='data/grid.dat', plot=False)
    pickle_save(g, f'data/graph.pkl')
