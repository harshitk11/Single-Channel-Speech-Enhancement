% This function calculates the segmental noisy signal to noise ratio.
% Input is signal power and noise power which is already divided into frames.
% Output is a vector. Each element of the vector is the NSNR of each frame.

function snr = nsnr(nsignal, noise)

% nsignal : power of the noisy signal
% noise : power of the estimated noise

x = size(nsignal);
framesize = x(1);
framenum = x(2);
snr = zeros(framenum,1);


for i = 1 : framenum
    spow = 0;
    npow = 0;
    for j = 1 : framesize
        spow = spow + nsignal(j,i);
        npow = npow + noise(j,i);  
    end
    snr(i) = 10*log10(spow/npow);
end
