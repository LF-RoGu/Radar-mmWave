clear all

Nsamples = 1500;

Xsaved = zeros(Nsamples, 2);
Zsaved = zeros(Nsamples, 1);

Psaved = zeros(Nsamples, 2);

for k=1:Nsamples
  z = GetSonar();      
  [pos vel P] = DvKalman(z);
  
  Xsaved(k, :) = [pos vel];
  Zsaved(k)    = z;

  Psaved(k,1:2) = [P(1,1) P(2,2)];
end


dt = 0.02;
t  = 0:dt:Nsamples*dt-dt;

fig = figure ;
left_color = [0 0 0]; % black
right_color= [0 0 1]; % blue
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
hold on
yyaxis left
plot(t, Zsaved(:), 'r.','markersize',20)
plot(t, Xsaved(:, 1),'k-','linewidth',4)
ylabel('position (m)')

yyaxis right
plot(t, Xsaved(:, 2),'b-','linewidth',4)
xlabel('time (s)'); ylabel('velocity (m/s)')
title('Position and Velocity from Noisy Sonar Measurements')
legend('Position, Raw Measurements','Position from Kalman Filter',...
    'Velocity from Kalman Filter')

set(gca,'fontsize',18); grid on
set(gcf, 'Position',  [10, 100, 1200, 500])

%% Extra things to plot
figure
plot(t,Psaved(:,1),'k')

figure
plot(t,Psaved(:,2),'b')

figure
t=0:dt:(Nsamples-1)*dt-dt;
vel_est = diff(Zsaved)/dt;
plot(t,vel_est,'r.','markersize',20)
hold on
vel_est2 = diff(Xsaved(:,1))/dt;
plot(t,vel_est2,'k-','linewidth',2)
plot(t,Xsaved(1:(end-1),2),'b','linewidth',4)
grid
xlabel('time (s)'); ylabel('velocity (m/s)')
legend('Finite Difference of Measurement',...
       'Finite Difference of Estimated Position',...
       'Kalman Filter Velocity Estimate');
set(gca,'fontsize',18); grid on



