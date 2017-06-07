function frame = frames(signal)
% SEGMENTATION OF DATA INTO FRAMES
% Sampling Frequency : 8kHz
% Window : Hanning
% Window Length: 256    (32ms)
% Window is shifted by 128 points.

% 256 point Hanning Window
L = 256;
w = hann(L,'periodic');
%wvtool(w);

orig = signal;
y = orig; 
% Creating the frames using the Hanning window
% Initializing the frame
frame = zeros(256,100);
i = 1;
for x = 1:128:(floor(length(y)/128)-1)*128
    frame(:,i) = w .* y(x:x+255,1);
    i = i+1;
end

% frame contains the frames of the signal after windowing

%-----------------------------------------------------------------------%
% Reconstructing the signal for verification
recon = zeros(length(orig),1);

sz = size(frame);
i = 1;
for j = 1:128:(sz(2)-1)*128
    recon(j:j+255) = recon(j:j+255) + frame(:,i);
    i = i+1;
end

% figure
% subplot(211)
% plot(orig);
% title('ORIGINAL SIGNAL');
% subplot(212);
% plot(recon);
% title('RECONSTRUCTED SIGNAL');
%-------------------------------------------------------------------------%
% Signal is divided into frames of size 256 points with a 50% overlapping.
% The sum of the windowed sequence adds back to the original sequence 
% (Refer paper 21: Spectral Substraction ORIGINAL). We can proceed with 
% this method.
%-------------------------------------------------------------------------%
end

