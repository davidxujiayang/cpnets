function [val] = eval_lag2D(eta,zeta,p_DG,i,j)

xn = linspace(-1,1,p_DG+1)'; 

val = lagrange(xn, i, eta).*lagrange(xn, j, zeta);
end

