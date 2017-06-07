load('C:\Users\admin\Documents\MATLAB\noise\babble.mat')
[p,q] = rat(8/19.98);
babbler = resample(babble,p,q);

audiowrite('noise_babbler.wav',babbler,8000);
audiowrite('noise_babble.wav',babble,19.98e3);

[stemp,ftemp] = audioread('noise_babbler.wav');
player = audioplayer(stemp,ftemp);
play(player)

% [stemp,ftemp] = audioread('noise_babble.wav');
% player = audioplayer(stemp,ftemp);
% play(player)