% Use the adaptive noise spectral estimation.
% Use the time-frequency filtering to reduce musical noise.
% NOISE ESTIMATION IS BASED ON PAPER 11

% Here parameters alpha(Over-estimation factor) and beta (Spectral Floor) have been introduced, which is made to change dynamically with
% the frames.


%% READING THE SIGNAL

cleansp1 = audioread('D:\Final_test\sp01.wav');        % clean speech
[rawsig1,fs1] = audioread('D:\Final_test\sp01_car_sn15.wav');  % raw signals
fs = 8000;
[p,q] = rat(fs/fs1);

% Resampling the signal to 8kHz
rawsig2 = resample(rawsig1,p,q);
cleansp2 = resample(cleansp1,p,q);

% While using wavesurfer to record the audio, wave surfer adds a preamble
% for the first 0.5 seconds (complete silence) which has to be skipped,
% else the algorithm will fail.

% rawsig = rawsig2(8000:length(rawsig2),1);
% cleansp = cleansp2(8000:length(rawsig2),1);

rawsig = rawsig2;
cleansp = cleansp2;
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
% getting the rough estimate of the clean speech, masking threshold will be
% calculated fromt the rough estimate of the clean speech.

% modified noise spectrum estimates
% = original estimates * overestimation factor
% Taking overeestimation factor = 1 

% Overestimation factor ov can be varied here depending on the need.
% Increasing the value of 'ov' will increase noise removal but it will 
% introduce distortion in the form of musical noise.

ov = 1;
noimag = noimag * ov;
% Noisy signal power : sigpow
% Noise power        : noipow

sigpow = sigmag.*sigmag;
noipow = noimag.*noimag;
% NSNR : Segmental Noisy signal to Noise Ratio
NSNR = nsnr(sigpow,noipow);

%%
%------------------------------------------------------------------------%
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

beta1 = 0.09;

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


% Uncomment the following line and comment line 177 if you don't want to
% use psychoacoustic masking.
% magtil = sqrt(denoipow); % magnitude of the rough estimate of the clean speech 

% clean_est is the estimate of the noiseless signal in the frequency
% domain. clean_est is passed as an argument to the 
% function speech_enhancement().
clean_est = sqrt(denoipow);


% Now we need to introduce masking based on psychoacoustic modelling.
% The psychoacoustic modelling uses the rough estimete of the clean speech.

%% CALCULATION OF MASKING THRESHOLD

% Calling the speech_enhancement function. This function calculate the
% noise masking threshold, and the values of alpha and beta on the basis of
% psychoacoustic modelling.

[alpha,beta,G] = speech_enhancement(clean_est,fs,noimag,rawsig);

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
% The following line calculates alpha and beta based on the masking model.
% (Refer paper 18)
% magtil = sqrt(maskpow);

% The following line calculates a perceptual filter based on the masking
% model. (Refer paper 26)
percepfilt = G.*sigmag;
magtil = percepfilt;

% At a time you can use any one of the above models (EITHER LINE 180 OR LINE 185).
% Line 185 yields better results. Line 180 is not yielding better results. 
% The interpretation of paper 18 (line 180) in this code might not be
% correct. Please check it.



%% TIME FREQUENCY FILTERING
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


%% ADDITIONAL NOISE SUPPRESSION
for p =1:length(nindex)
    k = nindex(p);
    T = sum(magtil(:,k)./noimag(:,k))/winleng;
    if T < 10^(-0.5)
        magtil(:,k) = magtil(:,k)*10^(-2);
    end
end

%% SMOOTHING
delta = 0.9;
for k=2:framenum
    magtil(:,k) = ( (1-delta)*magtil(:,k-1).^2+delta*magtil(:,k).^2 ).^(.5);
end

%% SYNTHESIS

% sigest: This is the enhanced speech.
% rawsig: This is the noisy speech.
% cleansp: This  is the clean speech.

sighat = magtil.*exp(1i*sigphase);
sigest_seg = real( ifft(sighat) );

sigest = 1.5 * real(syn(sigest_seg,overate));
% noiest = real(syn(noiest_seg,overate));

snr_out = 10*log(sum(sigest.^2) / sum((rawsig(1:dataleng) - sigest).^2))



% Plot
t = (1:dataleng)/fs;
figure
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

figure
subplot(3,1,1),specgram(cleansp(1:dataleng),256,fs);
title('Clean speech');
subplot(312),specgram(rawsig(1:dataleng),256,fs);
title('Noisy speech');
subplot(313),
specgram(sigest,256,fs);
title('Enhanced speech');

audiowrite('D:\Final_test\sp01_car_sn15_denoised.wav',sigest,fs);


% For showing the interpolation
noise = syn(noimag,overate);
signalf = syn(sigmag,overate);
figure
plot(signalf)
hold on
plot(noise)
title('NOISE CURVE (INTERPOLATION DONE IN THE FREQUENCY DOMAIN)');
legend('SIGNAL','INTERPOLATION OF THE NOISE');