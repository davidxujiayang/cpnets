function phi = lagrange(xn, j, x);
% computes values of jth lagrange function based on nodes
% in xn, at the x-values given in x

n = length(xn);
if (n == 1)
  phi = ones(size(x)); return;
end
xnj = xn([1:j-1,j+1:n]');
den = prod(xn(j)-xnj);
num = prod(repmat(x,1,n-1) - repmat(xnj',length(x),1), 2);
phi = num/den;
