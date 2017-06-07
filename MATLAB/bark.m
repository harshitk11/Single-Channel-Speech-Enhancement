% Critical band analysis is performed on the FFT power spectrum by adding
% up the energies in each critical band k

% Function takes the input as a 128 point signal and divides them into
% different 18 critical bands

% Output is band of length 18

function band = bark(signal)

k1 = signal(1:3).*signal(1:3);
k2 = signal(4:6).*signal(4:6);
k3 = signal(7:10).*signal(7:10);
k4 = signal(11:13).*signal(11:13);
k5 = signal(14:16).*signal(14:16);
k6 = signal(17:20).*signal(17:20);
k7 = signal(21:25).*signal(21:25);
k8 = signal(26:29).*signal(26:29);
k9 = signal(30:35).*signal(30:35);
k10 = signal(36:41).*signal(36:41);
k11 = signal(42:47).*signal(42:47);
k12 = signal(48:55).*signal(48:55);
k13 = signal(56:64).*signal(56:64);
k14 = signal(65:74).*signal(65:74);
k15 = signal(75:86).*signal(75:86);
k16 = signal(87:100).*signal(87:100);
k17 = signal(101:118).*signal(101:118);
k18 = signal(119:128).*signal(119:128);

k = zeros(1,18);
k(1) = sum(k1);
k(2) = sum(k2);
k(3) = sum(k3);
k(4) = sum(k4);
k(5) = sum(k5);
k(6) = sum(k6);
k(7) = sum(k7);
k(8) = sum(k8);
k(9) = sum(k9);
k(10) = sum(k10);
k(11) = sum(k11);
k(12) = sum(k12);
k(13) = sum(k13);
k(14) = sum(k14);
k(15) = sum(k15);
k(16) = sum(k16);
k(17) = sum(k17);
k(18) = sum(k18);

band = k';
end