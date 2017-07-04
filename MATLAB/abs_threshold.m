% This function calculates the Absolute Threshold (Refer to paper 28)

% The SPL curve is often referenced to the coding system by equating the lowest
% point to the energy in +/- 1 bit of signal amplitude. In other words, it
% is assumed that the playback level(volume control) on a typical decoder
% will be set such that the smallest possible output signal will be
% presented close to 0dB SPL.


function t_abs_bark = abs_threshold(sig)

len = size(sig);
framenum = len(2);

t_abs_bark = zeros(18,framenum);

% The 128th point corresponds to the 4kHz frequency
f = 1:31.25:4000;             % For getting a 128 point absolute threshold
t_abs_tmp = 3.64*power((f/1000),(-0.8))-6.5*exp(-0.6*power(((f/1000)-3.3),2))+power(10,-3)*power((f/1000),4);
% figure
% plot(t_abs_tmp);
for i=1:framenum
    offset = 20*log10(abs(min(sig(:,i))));
    % Final absolute threshold
    t_abs = t_abs_tmp + offset;

    % Converting the absolute threshold to the bark domain
    % Since the absolute threshold varies inside the critical band, the mean of
    % the values at the band edges is used.

    

    t_abs_bark(1,i) = mean([t_abs(1) t_abs(3)]);  
    t_abs_bark(2,i) = mean([t_abs(4) t_abs(6)]);
    t_abs_bark(3,i) = mean([t_abs(7) t_abs(10)]);
    t_abs_bark(4,i) = mean([t_abs(11) t_abs(13)]);
    t_abs_bark(5,i) = mean([t_abs(14) t_abs(16)]);
    t_abs_bark(6,i) = mean([t_abs(17) t_abs(20)]);
    t_abs_bark(7,i) = mean([t_abs(21) t_abs(25)]);
    t_abs_bark(8,i) = mean([t_abs(26) t_abs(29)]);
    t_abs_bark(9,i) = mean([t_abs(30) t_abs(35)]);
    t_abs_bark(10,i) = mean([t_abs(36) t_abs(41)]);
    t_abs_bark(11,i) = mean([t_abs(42) t_abs(47)]);
    t_abs_bark(12,i) = mean([t_abs(48) t_abs(55)]);
    t_abs_bark(13,i) = mean([t_abs(56) t_abs(64)]);
    t_abs_bark(14,i) = mean([t_abs(65) t_abs(74)]);
    t_abs_bark(15,i) = mean([t_abs(75) t_abs(86)]);
    t_abs_bark(16,i) = mean([t_abs(87) t_abs(100)]);
    t_abs_bark(17,i) = mean([t_abs(101) t_abs(118)]);
    t_abs_bark(18,i) = mean([t_abs(119) t_abs(128)]);
end
% 
% figure
% plot(barkplot(t_abs_bark));
% hold on
% plot(barkplot(t_abs_bark),'*');
end