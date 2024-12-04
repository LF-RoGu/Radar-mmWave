clear all

Nsamples = 1500;
Xsaved   = zeros(Nsamples, 1);
Xmsaved  = zeros(Nsamples, 1);


for k=1:Nsamples
  xm = GetSonar();
  x  = MovAvgFilter(xm);
  
  Xsaved(k)  = x;
  Xmsaved(k) = xm;
end


dt = 0.02;
t  = 0:dt:Nsamples*dt-dt;

figure
hold on
plot(t, Xmsaved, 'r.');
plot(t, Xsaved, 'k','linewidth',3);
legend('Measured', 'Moving average','location','northwest')
