% Drop me a mail at 'harshitk11@gmail.com' before proceeding with this
% section.

% Assuming Noise detection is done. We are now proceeding with the
% substractive type algorithm.

% Reading the signal
[orig,fs] = audioread('sp01_airport_sn15.wav');
%----------------------------------------------------------------------%
% Designing FIR filter for to increase discrimination between speech and
% noise signals.

% b = [1 -0.95];
% a = 1;
% fout = filter(b,a,orig);

fout = orig;

% figure;
% subplot(211);
% plot(orig);
% 
% title('ORIGINAL SIGNAL');
% subplot(212);
% plot(fout);
% title('FILTERED SIGNAL');

% Dividing the resultant signal into frames
frame = frames(fout);

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
% stem(f,abs(ft(:,100)));
% hold on
% xlabel('FREQUENCY');
% ylabel('MAGNITUDE');

%----------------------------------------------------------------------%

%% NOISE MASKING THRESHOLD CALCULATION
% Now,each of the frames of the signal is divided into bark scales.

len = size(ft);
% Array to store the bark scales of all the frames 
bband = zeros(18,len(2));   
for i = 1:len(2)
    bband(:,i) = bark(abs(ft(:,i)));
end

bband_db = 10*log10(bband);

% bband stores the bark scales of all the frames
% bband_db stores the bark scales in dB domain
%----------------------------------------------------------------------%

% Convolution with a spreading function to take into account masking
% between different frequency bands.

% We need to convolute bband with the spreading function. The result of
% this convolution is an excitation pattern 'e'

% Spreading function is used the same as in paper (23)

% Critical band number k
% Since sampling frequency = 8 kHz so there are 18 critical bands
%% DIRECT CONVOLUTION (METHOD 1)
% k = 1:18;
% 
% % Spreading function in dB : Sf_dB
% Sf_dB = 15.81 +7.5*(k+0.474)-17.5*(1+(k+0.474).^2).^0.5;  
% Sf = power(10,(Sf_dB/10));  
% 
% 
% figure
% plot(k,Sf_dB);
% title('Spreading function in dB');
% xlabel('Critical band k');
% ylabel('Sf_dB');
% 
% etemp = zeros(35,len(2));
% for i = 1:len(2)
%     etemp(:,i) = conv(bband(:,i),Sf);
% end
% 
% e = etemp(1:18,:);
% % e : spread spectrum

%% CONVOLUTION USING A TOEPLITZ MATRIX (METHOD 2)
% This method is preferred as this is faster and is used in the paper
% Critical band number k
% Since sampling frequency = 8 kHz so there are 18 critical bands

k = 1:18;
% Spreading function in dB : Sf_dB
Sf_dB = 15.81 +7.5*(k+0.474)-17.5*(1+(k+0.474).^2).^0.5;  
Sf = power(10,(Sf_dB/10));  

% Creating a toeplitz matrix using Sf

S = toeplitz(Sf);

% e: spread spectrum
% Multiplication with Toeplitz is equivalent to convolution.
e = S*bband;


%-----------------------------------------------------------------------%
% %% COMPUTATIONALLY LESS EXPENSIVE RELATIVE THRESHOLD OFFSET (OPTIMIZED)(METHOD 1)
% % Substraction of a relative threshold offset O(k) depending on the noise like
% % or tone like nature of the masker and the maskee
% 
% % Signal in a lower critical band is more tone like in nature while a
% % signal in a higher critical band is more noise like.
% % The relarive threshold is set approximately -(14.5 + k)dB in the lowest
% % frequency tande of 0-2.5 kHz. The relative threshold is raised gradually
% % at frequencies above 2.5 kHz and is finally frozen at a value of about
% % -18 dB.
% % REFER PAPER 24
% % Odb is in dB
% 
% % OFFSET IS SUBSTRACTED FROM THE SPREAD SPECTRUM IN THE dB DOMAIN
% Odb = zeros(18,1);
% Odb(1:9) = -(1:9) - 16;
% Odb(10:14) = -25;
% Odb(15) = -24;
% Odb(16) = -23;
% Odb(17) = -22;
% Odb(18) = -19;
% %Odb(19:25) = -18;

%% ACTUAL RELATIVE THRESHOLD OFFSET (METHOD 2)

% Calculating arithemetic and geometric mean of power spectrum in each
% frame 
% am = arithemetic mean (size = no. of frames i.e. for each frame there is
% 1 arithemetic mean)
% gm = geometric mean ( same as am )

am = zeros(len(2),1);
gm = zeros(len(2),1);

for i = 1:len(2)
    am(i) = mean(abs(ft(:,i)).*abs(ft(:,i)));
    gm(i) = geo_mean(abs(ft(:,i)).*abs(ft(:,i)));
end

% SFM : Spectral Flatness Measure (Refer paper 26)
sfm_db_min = -60;
sfm_db = 10*log10(gm./am);

% Tonality coefficient : alpha_sfm
alpha_sfm = zeros(len(2),1);
for i = 1:len(2)
    alpha_sfm(i) = min((sfm_db(i)/sfm_db_min),1);
end

% Relative threshold offset in dB : Odb
Odb = zeros(18,1);    %For a sampling frequency of 8kHz
for k = 1:18
    Odb(k) = alpha_sfm(k)*(14.5 + k) + (1 - alpha_sfm(k))*5.5; 
end

%%
figure
plot(Odb,'*');
ylabel('Relative Threshold (in dB)');
xlabel('Critical Band Number k');

Otemp = power(10,Odb/10);
O = Otemp(1:18);
%------------------------------------------------------------------------%
% r = e - O;
% Substraction of the relative threshold offset from the excitation
% pattern in the dB domain.
edb = 10*log10(e);
Odbrep = repmat(Odb,1,len(2));

% Need to check the validity of the next statement. Whether to use 'abs' or
% not
tdb = edb - abs(Odbrep);    % Masking threshold in dB   
%r = power(10,log10(e) - (Odb/10));

figure
plot(barkplot(bband_db(:,54)));
hold on
plot(barkplot(edb(:,54)));
hold on
plot(barkplot(tdb(:,54)));
hold on
plot(10*log10(abs(ft(:,54)).*abs(ft(:,54))));
legend('Bark band in dB','Spread spectrum in dB','Masking threshold in dB','Power of signal in dB');
title('CALCULATION OF PERCEPTUAL THRESHOLD');
ylabel('dB');
xlabel('FFT bins');
%------------------------------------------------------------------------%
%% RENORMALIZATION
% REFER TO PAPER 25 AND 26
% t : masking threshold offset
% Converting the spread threshold back to the bark domain
t = power(10, tdb/10);

