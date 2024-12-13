clc
clear
close all
% Settings
f0              = 76e9; % Center frequency. In case of CW: Frequency of radar
complexMixer    = 1; % Set this to one to add an Q channel 
NoiseLevel      = 0.01; % Noise Level
Fontsize        = 15;
% Target settings in cartesian coordinates
%          y
%          ^
%          |
%          |
%     Radar-----> x
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define physical properties of target
% Target speed in cartesian coordinates
vx = 5;
vy = 0;
% Target position in cartesian coordinates
x0 = 5;
y0 = 5;


% Receive (ADC) settings
ObservationTime = 0.1;
SamplingFrequency = 10e3;


speedOfLight = 299792458;
% See e.g. https://www.radartutorial.eu/11.coherent/co06.en.html
dopplerConstant = 2*f0/speedOfLight;
fprintf('f0 = %g GHz leads to %f Hertz per (m/sec)\n',f0/1e9,dopplerConstant)
%%%%%%%%%%%%%%%%
% Check direct sampling:
T  = 1/f0;
fprintf('T for direct sampling = %g nsec\n',T*1e9/2);

lambda = 0.3/(f0/1e9);
fprintf('Lambda is %g\n',lambda);
wavenumber = 2*pi/lambda;

dt = 1/SamplingFrequency;
timeVector = 0:dt:ObservationTime;
NumberOfSamples = length(timeVector);
if mod(NumberOfSamples,2) ~= 0
    % Ensure that we have an even number of samples
    % This is only for FFT frequency axis
    NumberOfSamples = NumberOfSamples + 1;
    timeVector = [timeVector timeVector(end)+dt];
end
fprintf('Number of samples: %d\n',NumberOfSamples);
RangeOverTime = getRangeOverTime(vx,vy,x0,y0,timeVector);
speed = mean(diff(RangeOverTime))/dt;
ExpectedDopplerFrequency = speed*dopplerConstant;
%%%%%%%%%%%%%%%%%%%%%%%
% Memory estimation
% Assume that each sample is float
% After FFT it will be complex 
MyMemory = NumberOfSamples*4;  
fprintf('Memory used: %g kb\n',MyMemory/1024)
subplot(2,2,1);
plot(timeVector,RangeOverTime);
xlabel('Time','FontSize',Fontsize)
ylabel('Radial distance','FontSize',Fontsize);
TravelingTime = 2*RangeOverTime/speedOfLight;
fprintf('Minimum traveling time in nsec: %g\n',1e9*min(TravelingTime))
fprintf('Maximum traveling time in nsec: %g\n',1e9*max(TravelingTime))
%ExpectedDopplerFrequency = RangeRate*2*f0/speedOfLight;
%fprintf('Expected doppler:%g Hz\n',ExpectedDopplerFrequency);
RetartedTime  = timeVector - TravelingTime;
% Calculate the base band component of the downmixed signal
% cos(a)*cos(b) = 0.5*(cos(a+b)+cos(a-b))
% --> cos(a+b) is filtered out
receiveSignal = cos(2*pi*f0*timeVector - 2*pi*f0*RetartedTime);
receiveSignal = receiveSignal + NoiseLevel*randn(size(receiveSignal));
if complexMixer == 1
    receiveSignal = receiveSignal + 1j*sin(2*pi*f0*timeVector - 2*pi*f0*RetartedTime);
    receiveSignal = receiveSignal + 1j*NoiseLevel*randn(size(receiveSignal));
end
subplot(2,2,2);
plot(timeVector,real(receiveSignal));
xlabel('time in seconds','FontSize',Fontsize)
ylabel('signal (a.u.)','FontSize',Fontsize)

subplot(2,2,3);
Rx = fft(receiveSignal);
df = 1/ObservationTime; % frequency resolution of FFT
f = 0:df:(NumberOfSamples-1)*df;
% Calculate normalized spectrum in dB
Rx = abs(Rx);
Rx = 20*log10(Rx);
Rx = Rx - max(Rx);
% Plot
plot(f,Rx)
line([ExpectedDopplerFrequency ExpectedDopplerFrequency],[min(Rx) 0])
xlabel('f in Hz','FontSize',Fontsize)
ylabel('normalized spectrum in dB','FontSize',Fontsize)
subplot(2,2,4);
f = SamplingFrequency/NumberOfSamples*(-NumberOfSamples/2:NumberOfSamples/2-1);
Rx = fftshift(Rx);
plot(f,Rx);
line([ExpectedDopplerFrequency ExpectedDopplerFrequency],[min(Rx) 0])
xlabel('f in Hz','FontSize',Fontsize)
ylabel('normalized spectrum in dB','FontSize',Fontsize)
title('After FFT shift','FontSize',Fontsize)

function RangeOverTime = getRangeOverTime(vx,vy,x0,y0,timeVector)
    x = x0 + vx*timeVector;
    y = y0 + vy*timeVector;
    RangeOverTime = sqrt(x.^2+y.^2);
end
