function [kspec_avg,espec_avg] = spectrax(Lx,Ly,nx,ny,Ufield)

f = 1.0;

for j = 1:ny
    
    if(j==1)
    [kspec_avg,espec_avg] = spectra1D(Lx,nx,squeeze(Ufield(:,j)),f);
    else
    [kspec_1,espec_1] = spectra1D(Lx,nx,squeeze(Ufield(:,j)),f);    
    
    kspec_avg = kspec_avg + kspec_1;
    espec_avg = espec_avg + espec_1;
        
    end
    
end

kspec_avg = kspec_avg/ny;
espec_avg = espec_avg/ny;

end

