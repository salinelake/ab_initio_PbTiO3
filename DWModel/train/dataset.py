# import deepmd.DeepPot as DP
from deepmd.infer import DeepDipole # use it to load your trained model
from time import time 
import ase
import ase.io
# from fse.systems import PTO
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
    ax.scatter(x, y, s=np.ones_like(x)*6)

    hist, pos = np.histogram(y, bins=nbins )
    ax_histy.barh(pos[:-1],hist,height=(pos[1]-pos[0]),align='edge',log=True)

    # ax_histy.hist(y, bins=nbins, orientation='horizontal')
    # ax.tick_params(axis="y")
    ax.tick_params(axis="x",pad=-15)
    ax_histy.tick_params(axis="y",labelleft=False)
    ax_histy.tick_params(axis="x",pad=-15, labelsize=12)
    return

def get_data(dataset, model, datatype='fullwc'):
    dipole_ref = []
    global_ref = []
    dipole_pred = []
    global_pred = []
    nframe = 0
    # dataset = []
    for folder in dataset:
        data_dir = os.path.join(folder,datatype)
        print('loading ', data_dir) 
        cells = np.load(os.path.join(data_dir,'set.000/box.npy'))
        coords = np.load(os.path.join(data_dir,'set.000/coord.npy'))
        atypes = np.loadtxt(os.path.join(data_dir,'type.raw')).astype(int)
        nframe += cells.shape[0]
        dipole_ref.append(np.load(os.path.join(data_dir,'set.000/atomic_dipole.npy')).reshape(-1,3))
        global_ref.append(np.load(os.path.join(data_dir,'set.000/dipole.npy')).reshape(-1,3))
        _dipole_pred = model.eval(coords, cells, atypes)
        dipole_pred.append(_dipole_pred.reshape(-1,3))
        global_pred.append(_dipole_pred.sum(-2).reshape(-1,3))

    dipole_ref = np.concatenate(dipole_ref)
    dipole_pred = np.concatenate(dipole_pred)
    global_ref = np.concatenate(global_ref)
    global_pred = np.concatenate(global_pred)
    print('#frame=',nframe)
    return global_ref, global_pred, dipole_ref, dipole_pred

trainset = [
        "../DFT_data/dataset/PTO.init/02.md/cubic/300K",
        "../DFT_data/dataset/PTO.init/02.md/tetra/300K",
        "../DFT_data/dataset/PTO.init/02.md/cubic/600K",
        "../DFT_data/dataset/PTO.init/02.md/tetra/600K",
        "../DFT_data/dataset/iter.000000/data.001",
        "../DFT_data/dataset/iter.000000/data.004",
        "../DFT_data/dataset/iter.000001/data.003",
        "../DFT_data/dataset/iter.000003/data.004",
        "../DFT_data/dataset/iter.000002/data.004",
        "../DFT_data/dataset/iter.000004/data.004",
        "../DFT_data/dataset/iter.000005/data.005",
        "../DFT_data/dataset/iter.000006/data.005",
        "../DFT_data/dataset/iter.000007/data.003",
        "../DFT_data/dataset/iter.000008/data.005"
]
testset = [
    '../DFT_data/PTO.test/T900',
    '../DFT_data/PTO.test/T1000',
    '../DFT_data/PTO.test/T1200',
]

## load all training data
model_path = 'final_model/dipole-compress.pb'
model = DeepDipole(model_path)
prefix =  'compress' 
t0 = time()
global_ref, global_pred, dipole_ref, dipole_pred = get_data(trainset, model,'deepdp')
t1 = time()
print('Model: {},Evaluating trainset cost {}s'.format(model_path,t1-t0))
dipole_error=dipole_pred - dipole_ref
test_global_ref, test_global_pred, test_dipole_ref, test_dipole_pred = get_data(testset, model,'deepdp')
test_dipole_error = test_dipole_pred-test_dipole_ref

#=============  lazy   alias ===================#
e_ref = (global_ref**2).sum(-1)**0.5
e_error = ((global_pred-global_ref)**2).sum(-1)**0.5
f_ref = dipole_ref
f_pred = dipole_pred
f_error = f_pred - f_ref
test_e_ref = (test_global_ref**2).sum(-1)**0.5
test_e_pred = (test_global_pred**2).sum(-1)**0.5
test_e_error = ((test_global_pred-test_global_ref)**2).sum(-1)**0.5
test_f_ref = test_dipole_ref
test_f_pred = test_dipole_pred
test_f_error = test_f_pred - test_f_ref
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
scatter_hist(e_ref, e_error, axes[0], hist_axes[0])
axes[0].scatter(test_e_ref, test_e_error, s=np.ones_like(test_e_ref)*6, label='Test')
axes[0].set_xlabel(r"$\|p^G_{DFT}\|$ [e$\AA$]",fontdict=font)
axes[0].set_ylabel(r"$\|\Delta p^G\|$ [e$\AA$]",fontdict=font)
# axes[0].set_ylim(bottom=-0.5)
#################################
scatter_hist(f_ref[:,0], f_error[:,0] , axes[1], hist_axes[1])
axes[1].scatter(test_f_ref[:,0], test_f_error[:,0], s=np.ones_like(test_f_ref[:,0])*2, label='Test')
axes[1].set_xlabel(r"$p^x_{DFT}$ [e$\AA$]",fontdict=font)
axes[1].set_ylabel(r"$\Delta p^x$ [e$\AA$]",fontdict=font)
#################################
scatter_hist(f_ref[:,1], f_error[:,1], axes[2], hist_axes[2])
axes[2].scatter(test_f_ref[:,1], test_f_error[:,1], s=np.ones_like(test_f_ref[:,0])*2, label='Test')
axes[2].set_xlabel(r"$p^y_{DFT}$ [e$\AA$]",fontdict=font)
axes[2].set_ylabel(r"$\Delta p^y$ [e$\AA$]",fontdict=font)
#################################
scatter_hist(f_ref[:,2], f_error[:,2], axes[3], hist_axes[3])
axes[3].scatter(test_f_ref[:,2], test_f_error[:,2], s=np.ones_like(test_f_ref[:,0])*2, label='Test')
axes[3].set_xlabel(r"$p^z_{DFT}$ [e$\AA$]",fontdict=font)
axes[3].set_ylabel(r"$\Delta p^z$ [e$\AA$]",fontdict=font)
#################################
titles = ['(a)','(b)','(c)','(d)']
for idx, title in enumerate(titles):
    axes[idx].set_title(title, loc = 'left',fontsize=14)
plt.tight_layout()
plt.savefig('./{}-validation.png'.format(prefix),dpi=300)
plt.close(fig)