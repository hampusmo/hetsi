# HetSI

HetSI is a sparse symbolic regression algorithm, specifically set to be used with heterogeneous parameter fields. To run the code, clone the repository and install the dependencies through Conda in environment.yml with:

```
conda env create -f environment.yml
```

Depending on your needs for CUDA acceleration the correct package for PyTorch might require modification. Please see the PyTorch website for more information. 

Note that the code is intended to be a proof of concept, not a full production package.

## Folders

The hetsi folder contains the main model and supporting tools. 

- data.py contains data loading tools
- diff.py contains functions for numerical differentiation.
- loss.py contains loss and regularization functions.
- model.py contains the main model and basis classes.
- networks.py contains neural network architectures.
- utils.py contains other utilities.

The notebooks folder contains Jupyter notebooks for generating noisy data (/data), as well as for calculating Algebraic Helmholtz Inversion and running PySINDy (/references). The code with PySINDy uses its own environment, see environment file in notebooks/references.

The scripts folder contains five examples of applying HetSI to magnetic resonance elastography data in the frequency domain. 
The input data used is publicly available but require manual download. 

## Running HetSI

To run HetSI, you perform the following steps:

1. Load the data.
2. Apply filters/general preprocessing
3. Calculate bases, standard bases can be placed in a list and supplied to the regressor class.
4. Calculate dynamic bases, these need to be instantiated into their specific classes, and passed as a list.
5. Set up predictor network.
6. Specify the regressor parameters.
7. Instantiate the Regressor.
8. Run the regressor.setup method.
9. Run the regressor.regress method.

Example codes for four different cases are included in "Scripts". Note that the dynamic bases can be composed in many different ways and require manual preprocessing.

## Output

The output from the method can be fetched with the regressor.full_pred method and returns tensors which are flattened along the non-fixed axes. The fixed axes are moved to the right and to restore the original structure you should reshape the "flexible" dimensions back to their original shape. The fixed axes maintain their original order and shape.

## Datasets

The data used in the scripts are available through BIOQIC Apps (https://bioqic-apps.charite.de), as well as the Brain Biomechanics Imaging Repository (BBIR) (https://www.nitrc.org/projects/bbir/). 

The following scripts use the following data for the examples:

- train_boxfem.py: FEM box simulation - four_target_phantom.mat (BIOQIC)
- train_boxmre.py: MRE phantom data - phantom_unwrapped.mat (BIOQIC)
- train_brainfem.py: FEM simulation of a human brain - BrainSimDisplacements.mat (BIOQIC)
- train_invivo.py: U01_UDEL_0001_01_v4 (BBIR)

## Citation

If you find this work useful or use it as a part of your work, please include a reference to this repository. Further publication details may follow soon.