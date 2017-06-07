% % Reading the signal
% [orig,fs] = audioread('sp03_car_sn15.wav');
frame = frames(orig);

%Computing the Fourier Transform of the frames
% N point fft
N = 256;
ft = fft(frame,N);
% Symmetric signal, so we'll take only the positive frequency side
ft = ft(1:128,:);

%Frequency axis
%----------------------------------------------------------------------%
df = fs/N;
f = df*(0:(N/2)-1);
%----------------------------------------------------------------------%
% figure
% plot(f,abs(ft(:,150)));
% hold on
% xlabel('FREQUENCY');
% ylabel('MAGNITUDE');

%----------------------------------------------------------------------%
% Designing FIR filter for to increase discrimination between speech and
% noise signals.

b = [1 -0.95];
a = 1;
fout = filter(b,a,orig);

% figure;
% subplot(211);
% plot(orig);
%
% title('ORIGINAL SIGNAL');
% subplot(212);
% plot(fout);
% title('FILTERED SIGNAL');

play_orig = audioplayer(orig,fs);
play_filt = audioplayer(fout,fs);

%----------------------------------------------------------------------%
% Generating a speech metric (Refer paper 22)
fout_mod = abs(fout);

% Decay constant Bs yielding a decay rate of 150ms
Bs = 0.9992;

% sm : signal metric
sm = zeros(length(fout_mod),1);
sm(1) = fout_mod(1);

for i = 2:length(fout_mod)
    if sm(i) > fout_mod(i)
        sm(i) = fout_mod(i);

    elseif sm(i) <= fout_mod(i)
        sm(i) = (1-Bs)*fout_mod(i) + Bs*sm(i-1);
    end
end

%----------------------------------------------------------------------%
%Creating time axis
t = (1/fs):(1/fs):length(orig)/fs;
figure
subplot(211)
stem(fout_mod);
title('Signal Metric');
subplot(212)
stem(sm);

%----------------------------------------------------------------------%
% Generating a noise metric for a purely noise signal

% Decay constant Bn yielding a decay rate of 150ms
Bn = 0.9922;

% sm : signal metric
nm = zeros(length(fout_mod),1);
nm(1) = fout_mod(1);

for i = 2:length(fout_mod)
    if nm(i) > fout_mod(i)
        nm(i) = fout_mod(i);
    elseif nm(i) <= fout_mod(i)
        nm(i) = (1-Bn)*fout_mod(i) + Bn*nm(i-1);
    end
end

% figure
% subplot(211)
% stem(fout_mod);
% subplot(212)
% stem(nm);

%---------------------------------------------------------------------%
% Generating a true noise metric

% Decay constant Bn yielding a decay rate of 150ms
Bt = 0.999975;

% sm : signal metric
tnm = zeros(length(fout_mod),1);
tnm(1) = nm(1);

for i = 2:length(fout_mod)
    if tnm(i) <= nm(i)
        tnm(i) = (1-Bt)*nm(i)+Bt*tnm(i-1);
    elseif tnm(i) > nm(i)
        tnm(i) = nm(i);
    end
end

figure
subplot(211)
plot(t,fout);
title('True Noise Metric');
subplot(212)
plot(t,tnm);

%----------------------------------------------------------------------%
amax = max(fout_mod);  %Largest allowable input signal
Ts = 1;           % Speech threshold
Tn = 1.414;       % Noise threshold
%----------------------------------------------------------------------%
%Speech/Silence Detection
segment = zeros(length(orig),1);
Tmin = power(10,log10(amax)-2);

for i=1:length(orig)
    if sm(i) > (Ts*tnm(i) + Tmin)
        segment(i) = 0.5;
    elseif sm(i) < (Tn*tnm(i) + Tmin)
        segment(i) = 0;
    elseif (Tn*tnm(i) + Tmin <= sm(i) && sm(i) <= Ts*tnm(i) + Tmin)
        segment(i) = segment(i-1);
    end

end


figure
plot(orig)
hold on
plot(segment)
title('Speech/Silence Detection');