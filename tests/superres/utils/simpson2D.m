function [inte] = simpson2D(datap,dx,dy)

Nx = size(datap,1);
Ny = size(datap,2);

temparr = zeros(1,Ny);

for i = 1:Ny
    temparr(i) = simpson1D(datap(:,i),dx);
end

inte = simpson1D(temparr,dy);

end

