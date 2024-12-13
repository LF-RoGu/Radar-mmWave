clear all


dt = 0.2;
t  = 0:dt:10;

Nsamples = length(t);

Xsaved = zeros(Nsamples, 3);
Zsaved = zeros(Nsamples, 1);

for k=1:Nsamples
  z = GetVolt();  
  [volt Cov Kg] = SimpleKalman2(z);
  
  Xsaved(k,:) = [ volt Cov Kg ];
  Zsaved(k) = z;
end


figure
plot(t, Xsaved(:,1), 'o-','linewidth',4)
hold on
plot(t, Zsaved, 'r:*','markersize',10,'linewidth',2) 
xlabel('time (s)'); ylabel('voltage (V)')
title('Voltage')
legend('Kalman Filter', 'Measurements')
set(gca,'fontsize',18); grid on

figure
plot(t, Xsaved(:,2), 'o-','linewidth',4)
xlabel('time (s)')
ylabel('P (error covariance)')
set(gca,'fontsize',18); grid on


figure
plot(t, Xsaved(:,3), 'o-','linewidth',4)
xlabel('time (s)')
ylabel('K (Kalman gain)')
set(gca,'fontsize',18); grid on
