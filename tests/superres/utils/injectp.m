function [coeff] = injectp(inp,outp,coeffi)

pos = zeros((outp+1)^2,2);
lagint = linspace(-1,1,outp+1);
coeff = zeros(1,(outp+1)^2);

for i = 1:(outp+1)
    for j = 1:(outp+1)
        index = (j-1)*(outp+1)+i;
        pos(index,1) = lagint(i);
        pos(index,2) = lagint(j);
    end
end


for i = 1:(inp+1)
    for j = 1:(inp+1)
        index = (j-1)*(inp+1)+i;
        coeff(1,:) = coeff(1,:) + coeffi(index)*eval_lag2D(pos(:,1),pos(:,2),inp,i,j)';
    end
end

end

