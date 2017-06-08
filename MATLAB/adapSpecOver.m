% Use the adaptive noise spectral estimation.
% Use the time-frequency filtering to reduce musical noise.
% NOISE ESTIMATION IS BASED ON PAPER 11

% Most of the code taken from noise_estimator_1.m. Here parameters alpha
% and beta have been introduced, which is made to change dynamically with
% the frames.


%% READING THE SIGNAL

cleansp1 = audioread('C:\Users\admin\Documents\MATLAB\sp21_train_sn10.wav');        % clean speech
[rawsig1,fs1] = audioread('C:\Users\admin\Documents\MATLAB\sp21_train_sn10.wav');  % raw signals
fs = 8000;
[p,q] = rat(fs/fs1);

% Resampling the signal to 8kHz
rawsig = resample(rawsig1,p,q);
cleansp = resample(cleansp1,p,q);

%% SEGMENTATION INTO FRAMES

winleng = 256;                          % Window length
overate = 0.5;                          % Overlapping rate

rawsig_seg = seg(rawsig,winleng,overate);    % segmented raw signals

framenum = size(rawsig_seg,2);          % # of frames
% Data length after segmentation
dataleng = winleng*(1-overate)*(framenum - 1) + winleng;

%% NOISY SPEECH SPECTRAL ESTIMATION

% sigmag = Signal Magnitude
% noimag = Noise Magnitude
% We are only concerned with the spectral magnitude.

sigfft = fft(rawsig_seg);       % fft of segmented signal
sigphase=zeros(size(sigfft));   % phase of the noisy speech 
for k=1:framenum
    sigphase(:,k) = angle(sigfft(:,k));
end
sigmag = abs(sigfft);           %% spectral magnitude

%% VOICE ACTIVITY DETECTION ( DETECTION OF NOISE SEGMENTS)
D = zeros(1,framenum);
order = 13;     %% LP order
for k=1:framenum
    x = rawsig_seg(:,k);
    ener = x'*x;
    D(k) = ener*( 1-zcr(x) )*( 1-lpe(x,order) );
end
D=D/max(D);
dthresh = 0.05;
nindex = find(D <= dthresh);
sindex = find(D > dthresh);

% ALGORITHM BASED ON PAPER 11

% Adaptive noise spectral estimation
noimag = zeros(winleng,framenum);   %% original noise magnitude estimates 
noimag(:,1) = sigmag(:,1);          
% Suppose the first frame only contains noise. 
alpha1 = 0.9;
% N_(k) = (1-alpha)*X_i(k) + alpha*N_i(k-1)
beta1 = 2;
% test if X_i(k) > beta*N_i(k-1)

for l = 1:winleng   %% for each freq bin ( fft length = window length)
    for k = 2:framenum  %% for each frame
        %sigmag(l,k);
        %beta*noimag(l,k-1);
        if sigmag(l,k) > beta1*noimag(l,k-1)
            noimag(l,k) = noimag(l,k-1);
        else
            noiest = (1-alpha1)*sigmag(l,k) + alpha1*noimag(l,k-1);
            noimag(l,k) = noiest;
        end
    end
end

% To reduce errror frame averaging is used
for k = 1:framenum
    noimag(:,k) = mean(noimag(:,k:min(k+10,framenum)),2);
end

%% ADAPTIVE SPECTRAL OVERSUBSTRACTION (BASIC MODEL)
% At this point we have an estimete of the noise. Now we need to do
% spectral substraction. We will vary the value of alpha and beta (as
% defined in paper 26) to get a rough estimate of the clean speech. After
% getting the rough estimate of the clean speech pschoacoustic masking will
% be used to remove the residual musical noise.

% NSNR : Segmental Noisy signal to Noise Ratio


% modified noise spectrum estimates
% = original estimates * overestimation factor

noimag = noimag * 1;
% Noisy signal power : sigpow
% Noise power        : noipow

sigpow = sigmag.*sigmag;
noipow = noimag.*noimag;

NSNR = nsnr(sigpow,noipow);


% Calculating the over-substraction factor : alpha1
alpha1 = zeros(framenum,1);
for i = 1:framenum
    if (NSNR(i) >= 20)
        alpha1(i) = 1;
    elseif (NSNR(i) >=(-6) && NSNR(i) < 20)
        alpha1(i) = 4 - (3/20)*NSNR(i);
    elseif (NSNR(i) < (-6))
        alpha1(i) = 4.9;
    end
end

% beta is the floor noise parameter.
% beta should be in the range 0.005 to 0.02
% Fixing the value of beta to 0.01

beta1 = 0.01;

% denoipow : denoised signal power
denoipow = zeros(winleng,framenum);
for i = 1:framenum
    for j = 1:winleng
        if (sigpow(j,i)-alpha1(i)*noipow(j,i)) > (beta1*noipow(j,i))
            denoipow(j,i) = sigpow(j,i) - alpha1(i)*noipow(j,i);
        else
            denoipow(j,i) = beta1 * noipow(j,i);
        end
    end
end

%magtil = sqrt(denoipow); % magnitude of the rogh estimate of the clean speech 

% Now we need to introduce masking based on psychoacoustic modelling.
%% PSYCHOACOUSTIC MASKING 
maskpow = zeros(winleng,framenum);
for i = 1:framenum
    for j = 1:winleng
        if (denoipow(j,i)-alpha(j,i)*noipow(j,i)) > (beta(j,i)*noipow(j,i))
            maskpow(j,i) = denoipow(j,i) - alpha(j,i)*noipow(j,i);
        else
            maskpow(j,i) = beta(j,i) * noipow(j,i);
        end
    end
end

magtil = sqrt(maskpow);

%% CODE COPIED FRON NOISE_ESTIMATOR_1


%% Time-frequency filtering
index = zeros(2,2);
a1 = 7;
a2 = 7;
b1 = 4;
b2 = 4;
lambda = 5;
m = 0;
for l = 1:b1:winleng
    for k = 1:b2:framenum
        regb = magtil( max(1,l-b1):min(winleng,l+b1), max(1,k-b2):min(framenum,k+b2));
        rega = magtil( max(1,l-a1):min(winleng,l+a1), max(1,k-a2):min(framenum,k+a2));
        pb = sum(sum(regb));
        pa = sum(sum(rega)) - pb;
        if pb >= lambda*pa
            m=m+1;
            index(m,1) = l;
            index(m,2) = k;
        end
    end
end

for n = 1:m
    freqbin = index(n,1);
    frameind = index(n,2);
    t1 = max(frameind-b2,1);
    t2 = min(frameind+b2,framenum);
    f1 = max(freqbin-b1,1);
    f2 = min(freqbin+b1,winleng);
    magtil(f1:f2,t1:t2) = zeros(length(f1:f2),length(t1:t2))*10^(-4);
end


%% Additional noise suppression
for p =1:length(nindex)
    k = nindex(p);
    T = sum(magtil(:,k)./noimag(:,k))/winleng;
    if T < 10^(-0.5)
        magtil(:,k) = magtil(:,k)*10^(-2);
    end
end

%% Smoothing
delta = 0.9;
for k=2:framenum
    magtil(:,k) = ( (1-delta)*magtil(:,k-1).^2+delta*magtil(:,k).^2 ).^(.5);
end

%% Synthesis

% sigest: This is the enhanced speech.
% rawsig: This is the noisy speech.
% cleansp: This  is the clean speech.

sighat = magtil.*exp(1i*sigphase);
sigest_seg = real( ifft(sighat) );
% noiest_seg = real( ifft(noimag) );
% for p =1:length(nindex)
%     k = nindex(p);
%     sigest_seg(:,k) = sigest_seg(:,k)*10^(-1);
% end

% lpccoef = lpc(sigest_seg,13);
% for k = 1:framenum
%     sigest2_seg(:,k) = filter([0,-lpccoef(k,2:end)],1,sigest_seg(:,k));
% end
% 
sigest = real(syn(sigest_seg,overate));
% noiest = real(syn(noiest_seg,overate));

snr_out = 10*log(sum(sigest.^2) / sum((rawsig(1:dataleng) - sigest).^2))



% Plot
t = (1:dataleng)/fs;
figure(1);
subplot(3,1,1), plot(t,cleansp(1:dataleng));
xlabel('Time (s)');
title('Clean speech');
subplot(3,1,2), plot(t,rawsig(1:dataleng));
xlabel('Time (s)');
title('Noisy speech');
subplot(3,1,3),
plot(t,sigest);
xlabel('Time (s)');
title('Enhanced speech');

figure(2)
subplot(3,1,1),specgram(cleansp(1:dataleng),256,fs);
title('Clean speech');
subplot(312),specgram(rawsig(1:dataleng),256,fs);
title('Noisy speech');
subplot(313),
specgram(sigest,256,fs);
title('Enhanced speech');
% soundsc(cleansp,fs);
% soundsc(rawsig,fs);
% pause;
% soundsc(sigest,fs);
audiowrite('C:\Users\admin\Documents\MATLAB\NOISY SIGNALS\buc_5db_denoised.wav',sigest,fs);


% For showing the interpolation
noise = syn(noimag,overate);
signalf = syn(magtil,overate);
figure
plot(signalf)
hold on
plot(noise)
title('NOISE CURVE (INTERPOLATION DONE IN THE FREQUENCY DOMAIN');
legend('NOISE CURVE','STFT of the noisy signal');