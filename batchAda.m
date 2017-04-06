


% create batches -- partition delta into 6 subtraces -> 256 x 6080
%batch = linspace(1, delta_size, 9); % 7 batches
batch = 0:4560:delta_size;


% each delta_i is 256 x 4560 -- 
delta_1 = delta(:,batch(1)+1:batch(2));
delta_2 = delta(:,batch(2)+1:batch(3));
delta_3 = delta(:,batch(3)+1:batch(4));
delta_4 = delta(:,batch(4)+1:batch(5));
delta_5 = delta(:,batch(5)+1:batch(6));
delta_6 = delta(:,batch(6)+1:batch(7));
delta_7 = delta(:,batch(7)+1:batch(8));
delta_8 = delta(:,batch(8)+1:batch(9));


