function Phi = basis(xn, x);
% evaluates basis functions at x

n = length(x);
order = length(xn)-1;
Phi = zeros(n, order+1);
for p=0:order,
  B = lagrange(xn, p+1,x);
  Phi(:,p+1) = B';
end