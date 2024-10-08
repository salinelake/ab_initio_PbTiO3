preprocess.py extracts atomic configurations from the dataset (../DPModel/dataset.tar.gz) used to train the Deep potential model. Wannier90 calculations are generated from extracted atomic configurations based on templates in ./template.

Wannier90 results are collected and dumped as datasets (./dataset.tar.gz) through scripts in ./postprocess. 

The datasets are then used to train the deep dipole model in ./train/final_model.

The productive model is ./train/final_model/dipole-compress.pb

./train/dataset.py plot the error distribution.