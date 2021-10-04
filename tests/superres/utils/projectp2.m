function [coeff] = projectp(inp,outp,coeffi,pos,weights)

basis_qd2 = zeros((outp+1)^2,size(pos,1));
basis_qd  = zeros((inp+1)^2,size(pos,1));

for i = 1:(inp+1)
    for j = 1:(inp+1)
        index = (j-1)*(inp+1)+i;
        basis_qd(index,:) = eval_lag2D(pos(:,1),pos(:,2),inp,i,j);
    end
end

for i = 1:(outp+1)
    for j = 1:(outp+1)
        index = (j-1)*(outp+1)+i;
        basis_qd2(index,:) = eval_lag2D(pos(:,1),pos(:,2),outp,i,j);
    end
end

size_MM2 = size(basis_qd2,1);
size_MM  = size(basis_qd,1);
MassMat2 = zeros(size_MM2,size_MM2);

%Assemble mass matrix
for i = 1:size_MM2
for j = 1:size_MM2
     MassMat2(i,j) = sum(basis_qd2(i,:).*basis_qd2(j,:).*weights);
end
end

Uqd = zeros(1,size(pos,1));
RU = zeros(size_MM2,1);

for i1 = 1:size_MM
            Uqd = Uqd + basis_qd(i1,:)*coeffi(i1);
end

for i1 = 1:size_MM2
    RU(i1) = sum(basis_qd2(i1,:).*Uqd.*weights);
end

coeff = injectp(outp,inp,(MassMat2\RU)');

end

