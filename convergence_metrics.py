import numpy as np
from plotting import scatter_coordinates
import matplotlib.pyplot as plt
'''
    Scripts to determine convergence and sampling in footprint simulations. Also contains some processing and sanity
    checks to run on new systems, e.g. to make sure the transformations are working
'''


def local_concentration_around_zero(traj, lip_1_sel, lip_2_sel, radius):
    '''
        Calculates the concentration of species around the trajectory centerpoint (usually the protein COM), within
        a given radius


    '''
    lip1_in_radius = np.sqrt((traj.xyz[:, lip_1_sel, :2] ** 2).sum(axis=2)) <= radius
    lip2_in_radius = np.sqrt((traj.xyz[:, lip_2_sel, :2] ** 2).sum(axis=2)) <= radius

    return lip1_in_radius.sum(axis=1) /  (lip1_in_radius.sum(axis=1) + lip2_in_radius.sum(axis=1))


def run_all_convergence_analyses(traj, indices, sel_of_interest="CDL2", sel_bg="POPC", radii=(6, 12), showframe=1000):
    scatter_coordinates(traj, indices, frame=showframe)

    for rad in radii:
        conc = local_concentration_around_zero(traj, indices[sel_of_interest], indices["POPC"], rad)
        plt.plot(traj.time / 1000, conc)
        plt.xlabel("time ns")
        plt.ylabel("local concentration of {}".format(sel_of_interest))
        plt.title("concentration within {:d} nm of protein COM".format(rad))
        plt.show()
