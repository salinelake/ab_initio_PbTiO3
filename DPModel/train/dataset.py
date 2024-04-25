# import deepmd.DeepPot as DP
from deepmd.infer import DeepPot as DP# use it to load your trained model,  DP2.0
import dpdata
from utility import *
E0 = -881.766

def add_multiaxes(fig, left, bottom, width, height, hist_width, spacing):
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, hist_width, height]
    ax = fig.add_axes(rect_scatter)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histy.set_xlabel("Count",fontdict=font)
    return ax, ax_histy

def scatter_hist(x,y, ax, ax_histy):
    nbins=20
    ax.scatter(x, y, s=np.ones_like(x)*6, label='Energy')

    hist, pos = np.histogram(y, bins=nbins )
    ax_histy.barh(pos[:-1],hist,height=(pos[1]-pos[0]),align='edge',log=True)

    # ax_histy.hist(y, bins=nbins, orientation='horizontal')
    # ax.tick_params(axis="y")
    ax.tick_params(axis="x",pad=-15)
    ax_histy.tick_params(axis="y",labelleft=False)
    ax_histy.tick_params(axis="x",pad=-15, labelsize=12)
    return

def get_data(dataset, dp):
    e_ref = []
    e_pred = []
    f_ref = []
    f_pred = []
    # dataset = []
    for data_dir in dataset:
        print('loading ', data_dir) 
        data = dpdata.LabeledSystem(data_dir,fmt='deepmd/raw')
        e_ref.append(data['energies'])
        atypes = data['atom_types']
        coords = data['coords'].reshape(-1, 3*atypes.shape[0])
        cells = data['cells'].reshape(-1, 9)
        forces = data['forces']
        f_ref.append(forces.reshape(-1,3))
        
        e, f, v = dp.eval(coords, cells, atypes)
        e_pred.append(e.flatten())
        f_pred.append(f.reshape(-1,3))
    e_ref = np.concatenate(e_ref).flatten()  / atypes.shape[0]
    e_pred = np.concatenate(e_pred).flatten()  / atypes.shape[0]
    e_ref = (e_ref - E0)*1000  #meV
    e_pred = (e_pred - E0)*1000  #meV
    f_ref = np.concatenate(f_ref)
    f_pred = np.concatenate(f_pred)
    return e_ref, e_pred, f_ref, f_pred

trainset = [
    "../DFT_data/PTO.init/02.md/cubic/300K/deepmd",
    "../DFT_data/PTO.init/02.md/cubic/600K/deepmd",
    "../DFT_data/PTO.init/02.md/cubic/900K/deepmd",
    "../DFT_data/PTO.init/02.md/tetra/300K/deepmd",
    "../DFT_data/PTO.init/02.md/tetra/600K/deepmd",
    "../DFT_data/PTO.init/02.md/tetra/900K/deepmd",
    "../DFT_data/iter.000000/02.fp/data.001",
    "../DFT_data/iter.000000/02.fp/data.004",
    "../DFT_data/iter.000001/02.fp/data.003",
    "../DFT_data/iter.000001/02.fp/data.002",
    "../DFT_data/iter.000001/02.fp/data.005",
    "../DFT_data/iter.000001/02.fp/data.000",
    "../DFT_data/iter.000002/02.fp/data.001",
    "../DFT_data/iter.000002/02.fp/data.004",
    "../DFT_data/iter.000003/02.fp/data.003",
    "../DFT_data/iter.000003/02.fp/data.001",
    "../DFT_data/iter.000003/02.fp/data.000",
    "../DFT_data/iter.000003/02.fp/data.004",
    "../DFT_data/iter.000004/02.fp/data.001",
    "../DFT_data/iter.000004/02.fp/data.004",
    "../DFT_data/iter.000005/02.fp/data.003",
    "../DFT_data/iter.000005/02.fp/data.001",
    "../DFT_data/iter.000005/02.fp/data.002",
    "../DFT_data/iter.000005/02.fp/data.005",
    "../DFT_data/iter.000006/02.fp/data.003",
    "../DFT_data/iter.000006/02.fp/data.001",
    "../DFT_data/iter.000006/02.fp/data.002",
    "../DFT_data/iter.000006/02.fp/data.005",
    "../DFT_data/iter.000006/02.fp/data.000",
    "../DFT_data/iter.000006/02.fp/data.004",
    "../DFT_data/iter.000007/02.fp/data.003",
    "../DFT_data/iter.000007/02.fp/data.000",
    "../DFT_data/iter.000008/02.fp/data.002",
    "../DFT_data/iter.000008/02.fp/data.005",
]
testset = [ 
    "../DFT_data/iter.000009/02.fp/data.001",
    "../DFT_data/iter.000009/02.fp/data.004",
    "../DFT_data/iter.000010/02.fp/data.001",
    "../DFT_data/iter.000010/02.fp/data.005",
    "../DFT_data/iter.000011/02.fp/data.001",
    "../DFT_data/iter.000011/02.fp/data.005"
    ]
extra_test = [
    "../DFT_data/_FP_TEST/4x4x4_test/T925_tetra/deepmd"
]
## load all training data
dp = DP('./final_model/model-compress.pb' )
prefix =  'model' 
e_ref, e_pred, f_ref, f_pred = get_data(trainset, dp)
f_error=f_pred-f_ref
test_e_ref, test_e_pred, test_f_ref, test_f_pred = get_data(testset, dp)
test_f_error=test_f_pred-test_f_ref
extra_e_ref, extra_e_pred, extra_f_ref, extra_f_pred = get_data(extra_test, dp)
extra_f_error=extra_f_pred-extra_f_ref
#============================================================# 
fig = plt.figure(figsize=(8, 8))
left = 0.15
width, height = 0.55, 0.18
hist_width, spacing = 0.18, 0.02
bottom_list = [0.76,0.52,0.28,0.04]
axes = []
hist_axes = []
for idx, bottom in enumerate(bottom_list):
    ax, ax_histy = add_multiaxes(fig, left, bottom, width, height, hist_width, spacing)
    axes.append(ax)
    hist_axes.append(ax_histy)
#################################
scatter_hist(e_ref,(e_pred-e_ref), axes[0], hist_axes[0])
axes[0].scatter(test_e_ref, test_e_pred-test_e_ref, s=np.ones_like(test_e_ref)*3, label='Test')
axes[0].scatter(extra_e_ref, extra_e_pred-extra_e_ref, s=np.ones_like(extra_e_ref)*6, label='Test',color='yellow')
axes[0].set_xlabel(r"$E_{DFT}-E_0$ [meV/atom]",fontdict=font)
axes[0].set_ylabel(r"$\Delta E$[meV/atom]",fontdict=font)
#################################
scatter_hist(f_ref[:,0], f_error[:,0] , axes[1], hist_axes[1])
axes[1].scatter(test_f_ref[:,0], test_f_error[:,0], s=np.ones_like(test_f_ref[:,0])*2, label='Test')
axes[1].scatter(extra_f_ref[:,0], extra_f_error[:,0], s=np.ones_like(extra_f_ref[:,0])*2, label='extra',color='yellow')
axes[1].set_xlabel(r"$F^x_{DFT}$ [eV/$\AA$]",fontdict=font)
axes[1].set_ylabel(r"$\Delta F_x$ [eV/$\AA$]",fontdict=font)
#################################
scatter_hist(f_ref[:,1], f_error[:,1], axes[2], hist_axes[2])
axes[2].scatter(test_f_ref[:,1], test_f_error[:,1], s=np.ones_like(test_f_ref[:,0])*2, label='Test')
axes[2].scatter(extra_f_ref[:,1], extra_f_error[:,1], s=np.ones_like(extra_f_ref[:,1])*2, label='extra',color='yellow')
axes[2].set_xlabel(r"$F^y_{DFT}$ [eV/$\AA$]",fontdict=font)
axes[2].set_ylabel(r"$\Delta F_y$ [eV/$\AA$]",fontdict=font)
#################################
scatter_hist(f_ref[:,2], f_error[:,2], axes[3], hist_axes[3])
axes[3].scatter(test_f_ref[:,2], test_f_error[:,2], s=np.ones_like(test_f_ref[:,0])*2, label='Test')
axes[3].scatter(extra_f_ref[:,2], extra_f_error[:,2], s=np.ones_like(extra_f_ref[:,2])*2, label='extra',color='yellow')
axes[3].set_xlabel(r"$F^z_{DFT}$ [eV/$\AA$]",fontdict=font)
axes[3].set_ylabel(r"$\Delta F_z$ [eV/$\AA$]",fontdict=font)
#################################
titles = ['(a)','(b)','(c)','(d)']
for idx, title in enumerate(titles):
    axes[idx].set_title(title, loc = 'left',fontsize=14)

plt.tight_layout()
plt.savefig('./{}-validation.png'.format(prefix),dpi=300)
plt.close(fig)