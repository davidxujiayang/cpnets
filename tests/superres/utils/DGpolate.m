function [Zfield,Xfield,Ufield] = DGpolate(Lx,Ly,nxi,nyi,p_DG,Umodes)

dx = Lx/nxi;
dz = Ly/nyi;

%Interpolating grid
Xfield = linspace(dx/2,Lx-dx/2,nxi);
Zfield = linspace(dz/2,Ly-dz/2,nyi);

%DG grid
Nx_dg = size(Umodes,1);
Nz_dg = size(Umodes,2);
dx_dg = Lx/Nx_dg;
dy_dg = Ly/Nz_dg;

Ufield = zeros(nxi,nyi);

%Create array of element index
for i = 1:nxi
    for j = 1:nyi
        
        xcord = Xfield(i);
        ycord = Zfield(j);
        
        idg = floor((xcord)/dx_dg)+1;
        jdg = floor((ycord)/dy_dg)+1;
        
        x_interpr = -1 + (xcord-(idg-1)*dx_dg)*2/dx_dg;
        z_interpr = -1 + (ycord-(jdg-1)*dy_dg)*2/dy_dg;
        
        for i2 = 1:(p_DG+1)
        for j2 = 1:(p_DG+1)
            nindex = (j2-1)*(p_DG+1)+i2;
            Ufield(i,j) = Ufield(i,j) + Umodes(idg,jdg,nindex)*eval_lag2D(x_interpr,z_interpr,p_DG,i2,j2);
        end
        end
    end
end



end

