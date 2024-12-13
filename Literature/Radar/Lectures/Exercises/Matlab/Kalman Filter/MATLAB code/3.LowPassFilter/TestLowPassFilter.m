clear all

Nsamples = 1500;
Xsaved   = zeros(Nsamples, 1);
Xmsaved  = zeros(Nsamples, 1);

alpha = 0.3 ;

for k=1:Nsamples
  xm = GetSonar();
  x  = LowPassFilter(xm,alpha);
  
  Xsaved(k)  = x;
  Xmsaved(k) = xm;
end


dt = 0.02;
t  = 0:dt:Nsamples*dt-dt;

%figure
hold on
plot(t, Xmsaved, 'r.','markersize',10);
plot(t, Xsaved, 'b','linewidth',3);
grid on
legend('Measured', 'Low Pass Filter','Location','northwest')
xlabel('time (seconds)'); ylabel('distance (m)');
title('Sonar to Measure Range (Distance)')
set(gca,'fontsize',18)