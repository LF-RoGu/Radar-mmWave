clear all

addpath Matlab/

Nsamples = 41500; % Number of data points
dt = 0.01; % seconds of time-step
psi = 0 ; % initial yaw guess
theta=0 ;
phi = 0 ;

load ArsGyro

w1 = wx ; w2 = wy ; w3 = wz ;
w  = [w1(:) w2(:) w3(:)] ;
tf = Nsamples*dt-dt ; % seconds
wt = 0:dt:tf;

y0=[1 0 0 0]'; % initial condition

[t,y]=ode45(@(t,y) EP_KDE(t,y,wt,w),[0 tf],y0);

b0=y(:,1);
b1=y(:,2);
b2=y(:,3);
b3=y(:,4);

bsq=b0.^2 + b1.^2 + b2.^2 + b3.^2 ;

figure
subplot(2,2,1)
plot(t,b0);
grid
xlabel('t');
ylabel('\beta_0');
set(gca,'Fontsize',20)

subplot(2,2,2)
plot(t,b1);
grid
xlabel('t');
ylabel('\beta_1');
set(gca,'Fontsize',20)

subplot(2,2,3)
plot(t,b2);
grid
xlabel('t');
ylabel('\beta_2');
set(gca,'Fontsize',20)

subplot(2,2,4)
plot(t,b3);
grid
xlabel('t');
ylabel('\beta_3');
set(gca,'Fontsize',20)


figure

plot(t,bsq);
grid
xlabel('t');
ylabel('\beta_0^2 + \beta_1^2 + \beta_2^2 + \beta_3^2');
set(gca,'Fontsize',20)

EulerSaved = zeros(length(t), 3);

for k=1:length(t)
    z = y(k,:) ;
    e = EP2Euler321(z);
    psi = e(1);
    theta=e(2);
    phi = e(3);
  
    EulerSaved(k, :) = [ psi theta phi ];
end 

PsiSaved   = EulerSaved(:, 1) * 180/pi;
ThetaSaved = EulerSaved(:, 2) * 180/pi;
PhiSaved   = EulerSaved(:, 3) * 180/pi;


figure

subplot(1,3,1)
hold on
plot(t, PhiSaved,'r')
title('Roll')
xlabel('time (s)'); ylabel('Roll, \phi (degrees)')
grid on
ylim(40*[-1 1]); xlim([0 420]);
set(gca,'fontsize',18)

subplot(1,3,2)
hold on
plot(t, ThetaSaved,'r')
title('Pitch')
xlabel('time (s)'); ylabel('Pitch, \theta (degrees)')
grid on
ylim(40*[-1 1]); xlim([0 420]);
set(gca,'fontsize',18)

subplot(1,3,3)
hold on
plot(t, PsiSaved,'r', 'linewidth',1);
title('Yaw')
xlabel('time (s)'); ylabel('Yaw, \psi (degrees)')
grid on
xlim([0 420]);
set(gca,'fontsize',18)

set(gcf, 'Position',  [10, 100, 1200, 500])