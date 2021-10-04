import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os


if not os.path.exists('images/contours'):
    os.mkdir('images/contours')

z_test = [650, 700, 750, 800, 850]
nx_list = [8, 16, 24, 32, 48, 64]
stage_list = ['dg', 'mlp_pred', 'cpmlp_pred']

for z in z_test:
    for nx in nx_list:
        for stage in stage_list:
            recon_fn = f'data/recon_{stage}{z}_{nx}.mat'

            data = loadmat(recon_fn)
            Nxdg = data['Nxdg'][0,0]
            Nzdg = data['Nzdg'][0,0]
            x_ref = data['x_ref'][0,]
            z_ref = data['z_ref'][0,]
            Nx_interp = data['Nx_interp'][0,0]
            Nz_interp = data['Nz_interp'][0,0]
            Basis_eval2  = data['Basis_eval2']
            if stage == 'dg':
                L2_solU2 = data['L2_solU2']
                kspec_x = data['kspec_P4']
                espec_x = data['espec_P4']
                kspec_y = data['k2spec_P4']
                espec_y = data['e2spec_P4']
            else:
                L2_solU2 = data['L2_solUNN']
                kspec_x = data['kspec_P4NN']
                espec_x = data['espec_P4NN']
                kspec_y = data['k2spec_P4NN']
                espec_y = data['e2spec_P4NN']
            
            kspec_xl = data['kspec_P2']
            espec_xl = data['espec_P2']
            kspec_yl = data['k2spec_P2']
            espec_yl = data['e2spec_P2']
                
            size_MM2 = data['size_MM2'][0,0]
            x_interp = data['x_interp'][0,]
            z_interp = data['z_interp'][0,]
            Usnap = data['Usnap']
            vmax = Usnap.max()
            vmin = Usnap.min()

            fig, ax = plt.subplots()

            us = []
            for i in range(Nxdg):
                for j in range(Nzdg):
                    x_interp2 = x_interp + x_ref[i]
                    z_interp2 = z_interp + z_ref[j]
                    U_proj = np.zeros((Nx_interp,Nz_interp))
                    for k in range(size_MM2):
                        U_proj = U_proj + Basis_eval2[:,:,k]*L2_solU2[-1,i,j,k]
                    us.append(U_proj.flatten())
                    ax.imshow(U_proj[:,::-1].T, extent=[x_interp2[0], x_interp2[-1], z_interp2[0], z_interp2[-1]], cmap='RdBu_r', vmax=vmax, vmin=vmin)

            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, np.pi)
            ax.set_axis_off()
            plt.savefig(f'images/recon_{stage}{z}_{nx}.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            us = np.asanyarray(us)
            np.savez(f'data/recon_{stage}{z}_{nx}.npz', us=us, kspec_x=kspec_x, espec_x=espec_x, kspec_y=kspec_y, espec_y=espec_y, kspec_xl=kspec_xl, espec_xl=espec_xl, kspec_yl=kspec_yl, espec_yl=espec_yl)

            if stage == 'dg':
                L2_solU2_L = data['L2_solU2_L']
                fig, ax = plt.subplots()

                for i in range(Nxdg):
                    for j in range(Nzdg):
                        x_interp2 = x_interp + x_ref[i]
                        z_interp2 = z_interp + z_ref[j]
                        U_proj = np.zeros((Nx_interp,Nz_interp))
                        for k in range(size_MM2):
                            U_proj = U_proj + Basis_eval2[:,:,k]*L2_solU2_L[-1,i,j,k]
                        ax.imshow(U_proj[:,::-1].T, extent=[x_interp2[0], x_interp2[-1], z_interp2[0], z_interp2[-1]], cmap='RdBu_r', vmax=vmax, vmin=vmin)

                ax.set_xlim(0, 2*np.pi)
                ax.set_ylim(0, np.pi)
                ax.set_axis_off()
                plt.savefig(f'images/contours/recon_low_{stage}{z}_{nx}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                        


