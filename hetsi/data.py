import numpy as np
import os
import re
import h5py as h5
from scipy.io import loadmat
from nibabel import load
from hetsi.utils import fftFreqFilter

def loadBioqicFEM(path):

    """Loads the FEM simulation of the box:
    1. Modifies the Bioqic dataset to the preferred shape: Freq, X, Y, Z, T.
    3. Precision kept at Float64 / Complex128, can be relaxed"""

    working_dir = os.getcwd()
    filename = "four_target_phantom.mat"

    try:
        os.chdir(path)

        phase_data = loadmat(filename)["u_ft"]

        os.chdir(working_dir)

        phase_data = np.transpose(phase_data, (4, 1, 0, 2, 3)).astype(np.complex128)

        return phase_data

    except:

        os.chdir(working_dir)
        raise Warning("Failed to load dataset")

def loadBioqic(path, freq_idx = 1, time_fft = True, version = "phantom_unwrapped.mat"):

    """Loads the unwrapped dataset and performs:
    1. Modifies the Bioqic dataset to the preferred shape: Freq, X, Y, Z, T, Dir.
    2. Zeroing all components not used.
    3. Precision kept at Float64 / Complex128, can be relaxed"""

    working_dir = os.getcwd()

    try:
        os.chdir(path)

        if version == "phantom_unwrapped.mat":

            with h5.File(version, "r") as file:

                phase_data = np.array([file["phase_unwrapped"]]).squeeze()

            print(phase_data.shape)
            phase_data = np.transpose(phase_data, (0, 4, 5, 3, 2, 1))

        elif version == "phantom_unwrapped_dejittered.mat":

            fdata = loadmat(version)
            phase_data = fdata["phase_unwrap_noipd"]

            phase_data = np.transpose(phase_data, (5, 1, 0, 2, 3, 4))

        os.chdir(working_dir)

        if time_fft:
            phase_data = fftFreqFilter(phase_data, t_dim = -2, freq_idxs = freq_idx)

        return phase_data

    except Exception as e:

        os.chdir(working_dir)
        raise Warning("Failed to load BIOQIC dataset", e)

def list_files(dir_path):

    """Get specification of files and subjects.
    
    dir_path: str - Path to top level folder, should only contain folders of unzipped subjects.
    
    Returns a dict of subjects containing a list of paths to [displacements (List), mask (List), ref_stiffness (List)]"""

    cwd = os.getcwd()

    try:
        os.chdir(dir_path)
    
        subjects_dict = {}

        for subject in os.listdir():
            subj_path = os.path.join(dir_path, subject)

            if re.fullmatch(r"U01_UDEL_[\d]{4}_01_v4", subject): #subj_comp.fullmatch(subject)

                subj_files = []
                subj_mask = []
                subj_ref = []

                for root, dirs, files in os.walk(subj_path):
                    
                    for file in files:
                        
                        if re.fullmatch(r"U01_UDEL_[\d]{4}_01_MRE_AP_[\d]{2}Hz_disp_[\w]{2}\.nii\.gz", file):

                            subj_files.append(os.path.join(root, file))
                        
                        elif re.fullmatch(r"U01_UDEL_[\d]{4}_01_MRE_AP_[\d]{2}Hz_props_shear_[\w]{4}\.nii\.gz", file):

                            subj_ref.append(os.path.join(root, file))
                        
                        elif re.fullmatch(r"U01_UDEL_[\d]{4}_01_MREreg_brainmask\.nii\.gz", file):

                            subj_mask.append(os.path.join(root, file))
                    
                    subjects_dict[subject] = [sorted(subj_files), subj_mask, sorted(subj_ref)]

        
    except:
        os.chdir(cwd)
        print("Could not find target folder.")
        pass

    finally:
        os.chdir(cwd)

    return subjects_dict

def load_subject(path_list: list[str], path_mask = None, path_reference = None):

    """Loads a single subject into arrays."""

    data_re = []
    data_im = []

    ref_re = []
    ref_im = []

    for path in path_list:
        print(path)
        
        if path.endswith("re.nii.gz"):
            data_re.append(load(path).get_fdata())
        
        elif path.endswith("im.nii.gz"):
            data_im.append(1j * load(path).get_fdata())
        
        else:
            print("Couldn't sort ", path, " into real or imaginary parts. \n")
    
    if path_mask is not None:
        mask = load(path_mask[0]).get_fdata()
    
    else:
        mask = None
    
    if path_reference is not None:
        for path in path_reference:
        
            if path.endswith("real.nii.gz"):
                ref_re.append(load(path).get_fdata())
            
            elif path.endswith("imag.nii.gz"):
                ref_im.append(1j * load(path).get_fdata())
            
            else:
                print("Couldn't sort ", path, " into real or imaginary parts. \n")

    data = np.stack(data_re, axis = 0) + np.stack(data_im, axis = 0)
    
    if path_reference is not None:
        ref = np.stack(ref_re, axis = 0) + np.stack(ref_im, axis = 0)
    
    else:
        ref = None

    return data, mask, ref