[orig,fs] = audioread('sp01_airport_sn0.wav');
y=specsub(orig,fs); 

audiowrite('denoised_signal.wav',y,fs);