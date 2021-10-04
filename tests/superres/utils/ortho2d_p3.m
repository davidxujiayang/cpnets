function [val] = ortho2d_p3(x,y,i,j)

val = ortho_p3(x,i).*ortho_p3(y,j);

end

