import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from plotting import MidPointNorm
from transformations import center_on_rings
from convergence_metrics import run_all_convergence_analyses
'''
    Analysis of flat systems for lipid footprints

    Trajectories (as of now) are protein and a marker head group, centered on the protein

    Need to rotate system so that the rings are aligned on the x axis before gridding and assignment

    Then put lipids back in box.

    Then grid up

'''


def load_gromacs_index(index_file):
    ''' Loads a gromacs style index file. Decrements all read indices by 1, as numbering starts at 1 in the files, but
        we'll be using these as array indices

        Parameters -
            index_file - path to a file
        Returns -
            index_dict - dictionary of index string : list of integer values
    '''
    with open(index_file, 'r') as fin:
        index_dict = {}
        curr_group = []
        curr_nums = []
        for line in fin:

            # check for opening and closing brackets
            if "[" in line and "]" in line:

                # add previous to dictionary only if one existed before - accounts for initial case
                if curr_group:
                    index_dict[curr_group] = curr_nums

                # reset group and index count
                curr_group = line.split("[", 1)[-1].split("]", 1)[0].strip()
                curr_nums = []
            elif curr_group:
                curr_nums += [int(i) - 1 for i in line.split()]    # decrement each one
        # one last time
        if curr_nums:
            index_dict[curr_group] = curr_nums
    return index_dict



def load_traj_and_indices(path, traj="step7_analysis.xtc", pdb="step7_analysis.pdb", index="step7_analysis.ndx"):
    ''' Convenience function for loading in mdtraj object

        Params:
            path - string with path to directory
        Returns:
            tuple of (mdtraj trajectory, index dictionary)
     '''
    return md.load_xtc(path + traj, top=path + pdb), load_gromacs_index(path + index)


def split_indices_by_leaflet(traj, indices, lipid_strings):
    '''
        Adds groups to an index dictionary. For each specified group in lipid_strings, adds an _upper and _lower
        group based on the z com
    '''
    all_lipid_indices = []
    for lipid in lipid_strings:
        all_lipid_indices += indices[lipid]
    com = traj.xyz[0, all_lipid_indices, 2].mean()
    z_upper_indices = list(np.where(traj.xyz[0, :, 2] > com)[0])
    z_lower_indices = list(np.where(traj.xyz[0, :, 2] < com)[0])

    for lip in lipid_strings:
        indices[lip + "_upper"] = list(set(indices[lip]) & set(z_upper_indices))
        indices[lip + "_lower"] = list(set(indices[lip]) & set(z_lower_indices))


def assign_type_to_grid(lipid_xy, grid_edges):
    '''
        lipid_xy : nframes * nlipids * 2 array of one type.
        grid_edges : n_bins + 1

    '''
    counts, _, _ = np.histogram2d(lipid_xy[:, :, 0].flatten(), lipid_xy[:, :, 1].flatten(), [grid_edges, grid_edges])
    return counts


def lipid_radius_std(coords):
    return np.std(np.sqrt((coords[:, :, :2] ** 2).sum(axis=2)), axis=0)


def filter_by_std(coords, cutoff):
    cutoff_mask = lipid_radius_std(coords) >= cutoff
    return coords[:, cutoff_mask, :]


if __name__ == "__main__":
    path = '/home/kevin/hdd/Projects/ATP/experimenting/flat/PC_CL_1mol/production/'
    grid_radius = 12
    grid_spacing = 0.5
    firstframe =  8000

    traj, index_dict = load_traj_and_indices(path)
    nframes = traj.xyz[firstframe:, :, :].shape[0]
    center_on_rings(traj, index_dict)
    split_indices_by_leaflet(traj, index_dict, ("POPC", "CDL2"))

    grid_centers = np.arange(-grid_radius, grid_radius + .0001, grid_spacing)
    grid_edges = np.arange(-grid_radius - grid_spacing, grid_radius + grid_spacing, grid_spacing) + grid_spacing / 2

    # lipid occupancy
    PC_upper_coords = filter_by_std(traj.xyz[firstframe:, index_dict["POPC_upper"], :], 1.5)
    CL_upper_coords = filter_by_std(traj.xyz[firstframe:, index_dict["CDL2_upper"], :], 1.5)
    PC_counts_upper   = assign_type_to_grid(PC_upper_coords, grid_edges)
    CL_counts_upper   = assign_type_to_grid(CL_upper_coords, grid_edges)

    total_counts_upper = PC_counts_upper + CL_counts_upper
    lipid_occupancy_upper = total_counts_upper / nframes
    CL_percentage_upper = 100 * CL_counts_upper / (PC_counts_upper + CL_counts_upper)

    # metrics of convergence
    run_all_convergence_analyses(traj, index_dict)
    plt.imshow(lipid_occupancy_upper)

    norm = MidPointNorm(midpoint=20)
    plt.imshow(CL_percentage_upper.T, cmap="seismic",
               vmin=0,
               vmax=60,
               interpolation="spline16",
               extent=(-12, 12, -12, 12),
               norm=norm)
    cbar = plt.colorbar()
    cbar.set_label(label="CL concentration")
    # plt.contour(upper_protein_contour.T, vmin=0, vmax=1, extent=(-12, 12, -12, 12), levels=1)
    plt.show()

    '''
    PC_counts_lower   = assign_type_to_grid(traj.xyz[firstframe:, index_dict["POPC_lower"],            :2], grid_edges)
    CL_counts_lower   = assign_type_to_grid(traj.xyz[firstframe:, index_dict["CDL2_lower"],            :2], grid_edges)
    total_counts_lower = PC_counts_lower + CL_counts_lower
    lipid_occupancy_lower = total_counts_lower / nframes
    CL_percentage_lower = 100 * CL_counts_lower / (PC_counts_lower + CL_counts_lower)
    CL_percentage_lower[prot_occupancy > 1] = 20
    CL_percentage_lower[lipid_occupancy_lower < 0.15] = 20


    lower_protein_contour = (prot_occupancy > 1) | (lipid_occupancy_lower < .15)

    plt.imshow(CL_percentage_lower.T, cmap="seismic",
               vmin=0,
               vmax=60,
               interpolation="spline16",
               extent=(-12, 12, -12, 12),
               norm=norm)
    cbar = plt.colorbar()
    cbar.set_label(label="CL concentration")
    plt.contour(lower_protein_contour.T, vmin=0, vmax=1, extent=(-12, 12, -12, 12), levels=1)
    plt.show()
    '''
