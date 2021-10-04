function [kspec_avg,espec_avg] = spectray(Lx,Ly,nx,ny,Ufield)

f = 1.0;

for i = 1:nx
    
    if(i==1)
    [kspec_avg,espec_avg] = spectra1D(Ly,ny,squeeze(Ufield(i,:))',f);
    else
    [kspec_1,espec_1] = spectra1D(Ly,ny,squeeze(Ufield(i,:))',f);    
    
    kspec_avg = kspec_avg + kspec_1;
    espec_avg = espec_avg + espec_1;
        
    end
    
end

kspec_avg = kspec_avg/nx;
espec_avg = espec_avg/nx;

end

