clear all

addpath Matlab/

IgnoreGravity = [] ; % don't ignore gravity
%IgnoreGravity = 1 ; % Ignore gravity

Nsamples = 41500; % Number of data points
EulerSaved = zeros(Nsamples, 3);

dt = 0.01; % seconds of time-step
psi = 0 ; % initial yaw guess
theta=0 ;
phi = 0 ;

for k=1:Nsamples
  [w1 w2 w3] = GetGyro();  % load gyro data w1, w2, w3
  A = eye(4) + dt*1/2*[ 0   -w1  -w2  -w3 ;
                        w1   0    w3  -w2 ;
                        w2  -w3   0    w1 ;
                        w3   w2  -w1   0  ];

  if isempty(IgnoreGravity) 
    [f1 f2] = GetAccel(); % load accelerometer data                 
    [phi theta] = EulerAccel(f1, f2); % radians
  end

  z = Euler3212EP([ psi theta phi ]');
  [psi theta phi] = EulerKalman(A, z); % radians
  
  EulerSaved(k, :) = [ psi theta phi ];

end 

PsiSaved   = EulerSaved(:, 1) * 180/pi;
ThetaSaved = EulerSaved(:, 2) * 180/pi;
PhiSaved   = EulerSaved(:, 3) * 180/pi;

t = 0:dt:Nsamples*dt-dt;

figure

subplot(1,3,1)
plot(t, PhiSaved)
title('Roll')
xlabel('time (s)'); ylabel('Roll, \phi (degrees)')
grid on
ylim(40*[-1 1]); xlim([0 420]);
set(gca,'fontsize',18)

subplot(1,3,2)
plot(t, ThetaSaved)
title('Pitch')
xlabel('time (s)'); ylabel('Pitch, \theta (degrees)')
grid on
ylim(40*[-1 1]); xlim([0 420]);
set(gca,'fontsize',18)

subplot(1,3,3)
plot(t, PsiSaved,'k', 'linewidth',1);
title('Yaw')
xlabel('time (s)'); ylabel('Yaw, \psi (degrees)')
grid on
xlim([0 420]);
set(gca,'fontsize',18)

set(gcf, 'Position',  [10, 100, 1200, 500])