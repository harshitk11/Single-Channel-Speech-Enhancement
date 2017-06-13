% % load('C:\Users\admin\Documents\MATLAB\noise\babble.mat')
% % [p,q] = rat(8/19.98);
% % babbler = resample(babble,p,q);
% % 
% % audiowrite('noise_babbler.wav',babbler,8000);
% % audiowrite('noise_babble.wav',babble,19.98e3);
% % 
% % [stemp,ftemp] = audioread('noise_babbler.wav');
% % player = audioplayer(stemp,ftemp);
% % play(player)
% % 
% % % [stemp,ftemp] = audioread('noise_babble.wav');
% % % player = audioplayer(stemp,ftemp);
% % % play(player)
% 
% x = 0:0.01:9*pi;
% y = sin(x);
% figure
% plot(x,y);
% title('original signal');
% 
% yf = fft(y,1024);
% figure
% plot(abs(yf));
% title('Fourier transform of original signal');
% 
% yfc = yf(1:513);
% 
% yfc_fl = fliplr(yfc(2:512));
% fl = [abs(yfc) abs(yfc_fl)];
% flf = ifft(fl);
% yr = ifft(yf);
% yrc = ifft(yfc);
% 
% figure
% plot(abs(yr));
% title('Reconstructed signal');
% 
% figure
% plot(abs(yrc));
% title('Reconstructed Signal with negative frequencies removed');
% 
% figure
% plot(abs(flf));
% title('Flipped and added');
% 
% 
% 
% 
% %--------------------------------------------------------------------%
% klm = fliplr((barkplot(tdb(:,54)))');
% slm = [(barkplot(tdb(:,54)))', klm];
% figure
% plot(10*log10(abs(ft(:,54)).*abs(ft(:,54))));
% figure
% plot(10*log10(abs(ft_temp(:,54)).*abs(ft_temp(:,54))));
% hold on
% plot(slm)

% Absolute Threshold

% The curve is often referenced to the coding system by equating the lowest
% point to the energy in +/- 1 bit of signal amplitude. In other words, it
% is assumed that the playback level(volume control) on a typical decoder
% will be set such that the smallest possible output signal will be
% presented close to 0dB SPL.

f = 1:31.25:4000;
t_abs_tmp = 3.64*power((f/1000),(-0.8))-6.5*exp(-0.6*power(((f/1000)-3.3),2))+power(10,-3)*power((f/1000),4);

offset = 20*log10(abs(min(rawsig)));
t_abs = t_abs_tmp+offset;
