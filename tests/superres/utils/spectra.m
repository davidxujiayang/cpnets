function [kspec,espec] = spectra(Lx,Ly,nx,ny,Ufield)

%Filter all planes
dkx = 2*pi/Lx;
dky = 2*pi/Ly;

k_gridx = (-nx/2):(nx/2-1);
k_gridy = (-ny/2):(ny/2-1);

k_gridxs = fftshift(k_gridx);
k_gridys = fftshift(k_gridy);

k_gridsx = k_gridxs*dkx;
k_gridsy = k_gridys*dky;

fftU = fftn(squeeze(Ufield))/(nx*ny);

kxmax = (nx/2)*dkx;
kymax = (ny/2)*dky;
kmax = ceil(sqrt(kxmax^2+kymax^2));

dk = max([dkx dky]);

kspec = dk*(0:ceil(kmax/dk));
espec = zeros(1,ceil(kmax/dk)+1);

for i = 1:nx
    for j = 1:ny
    kmag = sqrt(k_gridsx(i)^2+k_gridsy(j)^2); 
    kmag_int = floor(kmag/dk+0.5)+1;
    espec(kmag_int) = espec(kmag_int) + (0.5*abs(fftU(i,j))^2)/dk;
    end
end

end

