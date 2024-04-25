from collect_dipole import collect_wannier
from collect_wc import collect_wc

import os


wannier_data = [
    '/home/pinchenx/ferro/DeepWannier/PTO.init/02.md/cubic/300K',
    '/home/pinchenx/ferro/DeepWannier/PTO.init/02.md/tetra/300K',
    '/home/pinchenx/ferro/DeepWannier/PTO.init/02.md/cubic/600K',
    '/home/pinchenx/ferro/DeepWannier/PTO.init/02.md/tetra/600K',
    '/home/pinchenx/ferro/DeepWannier/iter.000000/data.001',
    '/home/pinchenx/ferro/DeepWannier/iter.000000/data.004',
    '/home/pinchenx/ferro/DeepWannier/iter.000001/data.003',
    '/home/pinchenx/ferro/DeepWannier/iter.000003/data.004',
    '/home/pinchenx/ferro/DeepWannier/cori_income/DeepWannier/iter.000002/data.004',
    '/home/pinchenx/ferro/DeepWannier/cori_income/DeepWannier/iter.000004/data.004',
    '/home/pinchenx/ferro/DeepWannier/cori_income/DeepWannier/iter.000005/data.005',
    '/home/pinchenx/ferro/DeepWannier/cori_income/DeepWannier/iter.000006/data.005',
    '/home/pinchenx/ferro/DeepWannier/cori_income/DeepWannier/iter.000007/data.003',
    '/home/pinchenx/ferro/DeepWannier/cori_income/DeepWannier/iter.000008/data.005',
    '/home/pinchenx/ferro/DeepWannier/PTO.test/T900',
    '/home/pinchenx/ferro/DeepWannier/PTO.test/T1000',
    '/home/pinchenx/ferro/DeepWannier/PTO.test/T1200',
]

make dataset for dipole model (unit: eA)
for folder in wannier_data:
    data_dir = os.path.join(folder,'wannier')
    outdir = os.path.join(folder,'deepdp')
    collect_wannier(data_dir, 'scf.out', 'PTO.wout', outdir, dp_version=2)


# # make dataset for wannier center model: output displacement of wannier center from its home atom (unit: A)
# for folder in wannier_data:
#     data_dir = os.path.join(folder,'wannier')
#     outdir = os.path.join(folder,'OPbwc')
#     collect_wc(data_dir, 'scf.out', 'PTO.wout', outdir, dp_version=2,type_map=['O','Pb','Ti'], wc_atype=['O','Pb'])
