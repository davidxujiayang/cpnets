function [Uout] = extendplane(Uinp)

ny = size(Uinp,1);
nx = size(Uinp,2);

Uinp = squeeze(Uinp);
Uinp = [Uinp Uinp(:,1)];
Uout = [Uinp;Uinp(1,:)];

end

