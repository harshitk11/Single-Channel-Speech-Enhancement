% Drop me a mail at 'harshitk11@gmail.com' before proceeding with this
% section, else you'll be doomed for life.

function [alpha,beta,G] = speech_enhancement(signal,fs,noimag)

% Assuming Noise detection is done. We are now proceeding with the
% substractive type algorithm.

% Reading the signal
% [orig,fs] = audioread('sp21_train_sn10.wav');
orig = signal;
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
ft_temp = ft;
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
framenum = len(2);
% Array to store the bark scales of all the frames 
bband = zeros(18,framenum);   
for i = 1:framenum
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
% etemp = zeros(35,framenum);
% for i = 1:framenum
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

am = zeros(framenum,1);
gm = zeros(framenum,1);

for i = 1:framenum
    am(i) = mean(abs(ft(:,i)).*abs(ft(:,i)));
    gm(i) = geo_mean(abs(ft(:,i)).*abs(ft(:,i)));
end

% SFM : Spectral Flatness Measure (Refer paper 26)
sfm_db_min = -60;
sfm_db = 10*log10(gm./am);

% Tonality coefficient : alpha_sfm
alpha_sfm = zeros(framenum,1);
for i = 1:framenum
    alpha_sfm(i) = min((sfm_db(i)/sfm_db_min),1);
end

% Relative threshold offset in dB : Odb
Odb = zeros(18,1);    %For a sampling frequency of 8kHz
for k = 1:18
    Odb(k) = alpha_sfm(k)*(14.5 + k) + (1 - alpha_sfm(k))*5.5; 
end

%%
% figure
% plot(Odb,'*');
% ylabel('Relative Threshold (in dB)');
% xlabel('Critical Band Number k');

Otemp = power(10,Odb/10);
O = Otemp(1:18);
%------------------------------------------------------------------------%
% r = e - O;
% Substraction of the relative threshold offset from the excitation
% pattern in the dB domain.
edb = 10*log10(e);
Odbrep = repmat(Odb,1,framenum);

% Need to check the validity of the next statement. Whether to use 'abs' or
% not
tdb = edb - abs(Odbrep);    % Masking threshold in dB   
%r = power(10,log10(e) - (Odb/10));



%% CALCULATION OF GAIN AFTER CONVOLUTION WITH THE SPREADING FUNCTION ( TO BE USED IN THE RENORMALIZATION STEP)

% bband*spreading_function -> e
% gain = e/bband

gain = e./bband;
%------------------------------------------------------------------------%
%% MODIFICATION IN MASKING THRESHOLD SINCE WE ARE USING AN ESTIMATE OF THE CLEAN SIGNAL (REFER PAPER 26 SECTION 4)

L = 10;
noipow = noimag.*noimag;
noiavg = (mean(noimag,2));  % Estimating the average noise power across all frames
%noiavg = noiavg_temp(1:128);

% corr : correction to be made in the noise masking threshold since we are
% using en estimate of the clean speech and not the actual clean speech.
corr_temp = zeros(length(noiavg),framenum);
% The last L frames are not evaluated.

for i = 1:framenum-L
    buf = zeros(length(noiavg),L);
    k = 1;
    for j = i:i+L-1
        buf(:,k) = abs(abs(noipow(:,j)-abs(noiavg.*noiavg))); 
        k=k+1;
    end
    foo = sum(buf);
    [foomax,fooind] = max(foo);
    corr_temp(:,i) = buf(:,fooind);
end

% Maxima is not taken for the last L frames. The difference between the
% noise power and the average value is directly assigned to the correction
% factor.

corr_temp(:,(framenum-L+1:framenum)) =  abs(abs(noipow(:,(framenum-L+1:framenum))-abs(noiavg.*noiavg)));

% Converting the correction factor to the bark domain.

corr_bband = zeros(18,framenum);   
for i = 1:framenum
    corr_bband(:,i) = bark(abs(sqrt(corr_temp(:,i))));  % Using sqrt() because the bark() function squares its input and then evaluates the output
end

% Correction : corr is in dB domain
corr = 10*log10(corr_bband);

% The modification of the threshold computation has to be made for high
% frequency domain (critical band >12)
tdb_mod = tdb;
tdb_mod(15:18,:) = tdb(15:18,:) - abs(corr(15:18,:));
% tdb_mod = tdb - abs(corr);


%------------------------------------------------------------------------%
%% RENORMALIZATION (TO BE USED ONLY WHEN CALCULATING THE VALUE OF ALPHA AND BETA)(NOT TO BE USED WITH PERCEPTUAL FILTER)
% REFER TO PAPER 25 AND 26
% t : masking threshold offset
% Converting the spread threshold back to the bark domain

% Since the spreading function increases the energy estimates in each band,
% it can be compensated at the renormalization stage "by multiplying each
% t(i) by the inverse of the energy gain, assuming a uniform energy of 1 in
% each band"



t = power(10, tdb_mod/10);
t_nom = t./gain;
tdb_nom = 10*log10(t_nom);

%% FACTORING IN THE VALUES OF ABSOLUTE THRESHOLD
tdb_abs_bark = abs_threshold(signal);
tf_bark = zeros(18,framenum);
for i = 1:framenum
    tf_bark(:,i) = max(tdb_abs_bark,tdb_nom(:,i));
end

figure
plot(barkplot(bband_db(:,54)));
hold on
plot(barkplot(edb(:,54)));
hold on
plot(barkplot(tdb(:,54)));
hold on
plot(10*log10(abs(ft(:,54)).*abs(ft(:,54))));
hold on
plot(barkplot(tdb_mod(:,54)),'o');
hold on
plot(barkplot(tdb_nom(:,54)),'*');
hold on
plot(barkplot(tf_bark(:,54)),'-');
hold on
legend('Bark band in dB','Spread spectrum in dB','Masking threshold in dB','Power of signal in dB','Modified threshold','Renormalized threshold');
title('CALCULATION OF PERCEPTUAL THRESHOLD');
ylabel('dB');
xlabel('FFT bins');



%% INCORPORATING THE MASKING MODELS IN ALPHA AND BETA


% Converting the bark domain from 128 points to 256 points, so that the
% masking can be applied to the entire 256 point fft. 

% Comment the following line if you don't want to use the modification in
% the threshold calculation.
%  t_temp = tdb_mod; 
% t_temp = tdb_nom;
t_temp = tf_bark;
tdb_f = zeros(256,framenum);

for i = 1:framenum
    klm = fliplr((barkplot(t_temp(:,i)))');
    tdb_f(:,i) = [(barkplot(t_temp(:,i)))', klm]';
end

t_f = power(10,(tdb_f/10));
% figure
% plot(10*log10(abs(ft_temp(:,54)).*abs(ft_temp(:,54))));
% hold on
% plot(tdb_f(:,54));
%% PERCEPTUAL WEIGHING FILTER (DO NOT USE WITH CALCULATION OF ALPHA AND BETA)
winleng = 256;
G = zeros(winleng,framenum);
one = ones(winleng,1);



for i = 1:framenum
    footemp = t_f(:,i)./ noiavg;
    footempa = [footemp one];
    G(:,i) =  min(footempa,[],2); 
end
%% CALCULATION OF ALPHA AND BETA
% Calculating the values of alpha and beta using the masking parameters

alpha = zeros(256,framenum);
beta = zeros(256,framenum);

for i = 1:framenum
    [alpha(:,i), beta(:,i)] = noise_mask(t_f(:,i));
end
end