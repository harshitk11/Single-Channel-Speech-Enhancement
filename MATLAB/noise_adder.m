clear all

%% LOADING THE NOISE
load('C:\Users\admin\Documents\MATLAB\noise\babble.mat')
[p,q] = rat(8/19.98);
% Downsampling the noise to 8kHz
noise =  resample(babble,p,q);

audiowrite('C:\Users\admin\Documents\MATLAB\noise\babble.wav',noise,8000);
% audiowrite('noise_babble.wav',babble,19.98e3);

[stemp,ftemp] = audioread('C:\Users\admin\Documents\MATLAB\noise\babble.wav');
player = audioplayer(stemp,ftemp);
play(player)
noisepow = bandpower(noise);

%% LOADING THE SIGNAL
[rawsig1,fs1] = audioread('C:\Users\admin\Documents\MATLAB\tata_downsampled.wav');  % raw signals
fs = 8000;
[p,q] = rat(fs/fs1);

% Downsampling the signal to 8kHz
rawsig = resample(rawsig1,p,q);
sigpow = bandpower(rawsig);
%% ADDING THE SIGNAL AND THE NOISE

% apriori SNR
asnr = 10*log(sum(rawsig.^2) / sum(noise.^2))

%noisy_signal = rawsig + noise(1:length(rawsig));

noisy_signal = [zeros(300000,1)' rawsig(300001:length(rawsig))']' + noise(1:length(rawsig));
audiowrite('C:\Users\admin\Documents\MATLAB\NOISY SIGNALS\buc_20db.wav',noisy_signal,8000);
 
[stemp,ftemp] = audioread('C:\Users\admin\Documents\MATLAB\NOISY SIGNALS\buc_20db.wav');
player = audioplayer(stemp,ftemp);
play(player);
