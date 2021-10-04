function [inte] = simpson1D(datap,dx)

N = max(size(datap));

inte = 0;

inte = inte + datap(1);
inte = inte + datap(N);
inte = inte + 2*sum(datap(3:2:N-1));
inte = inte + 4*sum(datap(2:2:N-1));
inte = inte*dx/3.0;

end

