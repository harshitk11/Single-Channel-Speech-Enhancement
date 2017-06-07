% Function to plot the threshold vs basrk scales.
% This function converts the bark scales back to the FFt bins.

function bplot = barkplot(signal)

out = zeros(128,1);

out(1:3) = signal(1);
out(4:6) = signal(2);
out(7:10) = signal(3);
out(11:13) = signal(4);
out(14:16) = signal(5);
out(17:20) = signal(6);
out(21:25) = signal(7);
out(26:29) = signal(8);
out(30:35) = signal(9);
out(36:41) = signal(10);
out(42:47) = signal(11);
out(48:55) = signal(12);
out(56:64) = signal(13);
out(65:74) = signal(14);
out(75:86) = signal(15);
out(87:100) = signal(16);
out(101:118) = signal(17);
out(119:128) = signal(18);

bplot = out;

end