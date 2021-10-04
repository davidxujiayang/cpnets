function [val] = ortho_p3(x,i)

if(i==1)
val = ones(size(x));
elseif(i==2)
val = x;
elseif(i==3)
val = 0.5*(3*x.*x-1);
elseif(i==4)
val = 0.5*(5*x.*x.*x-3*x);    
end

