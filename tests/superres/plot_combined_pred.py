import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(12.5, 2.5), dpi=300)
ax = []
w0 = 20
dx0 = 1
w1 = 20
dx1 = 4
dy = 1
h0 = 20
W = w0*4+dx0*3+dx1+w1
H = h0*2+dy

z0 = 650
Nx0 = 8
z1 = 750
Nx1 = 16

titles = ['Input (P=1)', 'Truth', 'MLP', r'$\bf{CP}$'+'-'+r'$\bf{MLP}$']
for i, case in enumerate(['low_dg', 'dg', 'mlp_pred', 'cpmlp_pred']):
    ax.append(plt.subplot2grid((H, W), (0, (w0+dx0)*i), rowspan=h0, colspan=w0))
    fn = f'images/contours/recon_{case}{z0}_{Nx0}.png'
    img = plt.imread(fn)
    ax[-1].imshow(img)
    ax[-1].set_title(titles[i])
    ax[-1].set_yticklabels([])
    ax[-1].set_xticklabels([])
    ax[-1].set_yticks([])
    ax[-1].set_xticks([])

    ax.append(plt.subplot2grid((H, W), (h0+dy, (w0+dx0)*i), rowspan=h0, colspan=w0))
    fn = f'images/contours/recon_{case}{z1}_{Nx1}.png'
    img = plt.imread(fn)
    ax[-1].imshow(img)
    # ax[-1].set_title(titles[i])
    ax[-1].set_yticklabels([])
    ax[-1].set_xticklabels([])
    ax[-1].set_yticks([])
    ax[-1].set_xticks([])

ax[0].set_ylabel(rf'$z^+={z0}$'+'\n'+rf'$L=\pi/{Nx0//2}$')
ax[1].set_ylabel(rf'$z^+={z1}$'+'\n'+rf'$L=\pi/{Nx1//2}$')

ax.append(plt.subplot2grid((H, W), (0, (w0+dx0)*4-dx0+dx1), rowspan=h0, colspan=w1))
data = np.load(f'data/recon_dg{z0}_{Nx0}.npz')
ax[-1].loglog(data['kspec_xl'][0,], data['espec_xl'][0,], f'k--', linewidth=0.8, label='P=1')
ax[-1].loglog(data['kspec_x'][0,], data['espec_x'][0,], f'k-', linewidth=0.8, label='Truth')
data = np.load(f'data/recon_mlp_pred{z0}_{Nx0}.npz')
ax[-1].loglog(data['kspec_x'][0,], data['espec_x'][0,], f'b-', linewidth=0.8, label='MLP')
data = np.load(f'data/recon_cpmlp_pred{z0}_{Nx0}.npz')
ax[-1].loglog(data['kspec_x'][0,], data['espec_x'][0,], f'r-', linewidth=0.8, label=r'$\bf{CP}$'+'-'+r'$\bf{MLP}$')

# ax[-1].legend(ncol=1, prop={'size': 8}, bbox_to_anchor=(1.15, 1.3))
ax[-1].legend(ncol=1, handlelength=1, borderpad=0.1, labelspacing=0.2, bbox_to_anchor=(0.6, 1.3))

ax[-1].set_title(r'Spectra', loc='left')
ax[-1].set_ylabel(r'$e_x$', rotation=90, fontsize=12)
ax[-1].yaxis.set_label_coords(-0.1,-0.01)
ax[-1].set_xlim([0.5,40])
ax[-1].set_ylim([2e-5,1])
ax[-1].grid()
ax[-1].tick_params(axis="y",direction="in")
ax[-1].tick_params(axis='both', which='major', labelsize=6)
ax[-1].set_xticklabels([])

ax.append(plt.subplot2grid((H, W), (h0+dy, (w0+dx0)*4-dx0+dx1), rowspan=h0, colspan=w1))
data = np.load(f'data/recon_dg{z1}_{Nx1}.npz')
ax[-1].loglog(data['kspec_xl'][0,], data['espec_xl'][0,], f'k--', linewidth=0.8)
ax[-1].loglog(data['kspec_x'][0,], data['espec_x'][0,], f'k-', linewidth=0.8)
data = np.load(f'data/recon_mlp_pred{z1}_{Nx1}.npz')
ax[-1].loglog(data['kspec_x'][0,], data['espec_x'][0,], f'b-', linewidth=0.8)
data = np.load(f'data/recon_cpmlp_pred{z1}_{Nx1}.npz')
ax[-1].loglog(data['kspec_x'][0,], data['espec_x'][0,], f'r-', linewidth=0.8)
ax[-1].tick_params(axis='both', which='major', labelsize=6)
# ax[-1].legend()
ax[-1].set_xlabel(r'$k_x$', labelpad=-5, fontsize=12)
ax[-1].set_xlim([0.5,40])
ax[-1].set_ylim([1e-5,2])
ax[-1].grid()
ax[-1].tick_params(axis="y",direction="in")


plt.savefig(f'images/compare.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.close()