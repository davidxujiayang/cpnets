function [kspec,espec] = spectra1D(Lx,nx,Ufield,f)

%Filter all planes
dkx = 2*pi/Lx;

k_gridx = (-nx/2):(nx/2-1);

k_gridxs = fftshift(k_gridx);

k_gridsx = k_gridxs;

fftU = fftn(squeeze(Ufield))/nx;

kxmax = (nx/2);
kmax = ceil(sqrt(kxmax^2));

dk = 1.0*f;

kspec = dk*(0:ceil(kmax/dk));
espec = zeros(1,ceil(kmax/dk)+1);

for i = 1:nx
    kmag = sqrt(k_gridsx(i)^2); 
    kmag_int = floor(kmag/dk+0.5)+1;
    espec(kmag_int) = espec(kmag_int) + (0.5*abs(fftU(i))^2)/dk;
end

kspec = kspec*dkx;

end

