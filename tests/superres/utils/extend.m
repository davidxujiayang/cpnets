function [Uplane2] = extend(Uplane)
Nx = size(Uplane,1);
Uplane(Nx+1,:) = Uplane(1,:);
Ny = size(Uplane,2);
Uplane(:,Ny+1) = Uplane(:,1);
Uplane2 = Uplane;
end

