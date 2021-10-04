function [Ufieldfil] = filterles(Lx,Ly,nx,Fsize,Ufield)

%Filter all planes
k_cut = pi/Fsize;
dkx = 2*pi/Lx;
dky = 2*pi/Ly;

k_grid = (-nx/2):(nx/2-1);
k_grids = fftshift(k_grid);

k_gridsx = k_grids*dkx;
k_gridsy = k_grids*dky;

fftU = fftn(squeeze(Ufield));

for i = 1:nx
    for j = 1:nx
    kmag = sqrt(k_gridsx(i)^2+k_gridsy(j)^2); 
    if(kmag>k_cut)
        fftU(i,j) = 0.0;
    end
    end
end

Ufieldfil = ifftn(fftU);
end

