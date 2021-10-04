function [pos,weights] = eval_quad2D(p_DG) %val pos and val2 weights

[x_quad,w_quad]=lgwt(2*p_DG+1,-1,1);

qd_size = size(x_quad,1);
qd = qd_size*qd_size;

pos = zeros(qd,2); %Eta zeta weights

for i = 1:qd_size
    for j = 1:qd_size
        index = (j-1)*qd_size+i;
        pos(index,1) = x_quad(i);
        pos(index,2) = x_quad(j);
        weights(index)  = w_quad(i)*w_quad(j);
    end
end

end

