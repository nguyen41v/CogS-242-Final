x = [0:0.0001:4];
y = (1./x) .* (1 - erf(1./sqrt(x)));
%plot(x,y);

h_mean = 16;
increment = 0.1;
max_std_dev = 25;
std_dev = [0:increment:max_std_dev];
%error_rate = zeros(max_std_dev/increment);

max_diff = 50;
difficulty = [0: increment: max_diff];
k = length(difficulty);
error_rate = zeros(k,1);
learning_rate = zeros(k,1);
for diff_i = 1:k
    %error_rate(diff_i,:) = normcdf([-inf, 0], h_mean, std_dev[diff_i]));
    gaussian_curve  = (normcdf([-inf, 0], difficulty(diff_i), h_mean));
    
    %disp(difficulty(diff_i));
    error_rate(diff_i,:) = gaussian_curve(2);
    pdf_curve  = (normpdf([-inf, 0], difficulty(diff_i), h_mean));
    learning_rate(diff_i,:) = pdf_curve(2) * difficulty(diff_i);
    %disp(error_rate(diff_i,:));
end

% Fig 1B
%f1 = figure;
%plot(difficulty, error_rate);
%disp(error_rate);
f2 = figure;
%plot(difficulty, learning_rate);
% length(error_rate)
% length(difficulty)
% difficulty1 = [0: increment: max_diff - increment];
% dERdB = -diff(error_rate)./diff(difficulty);
% length(dERdB)
% length(difficulty)
% dER = gradient(error_rate, learning_rate);
%disp(dERdB);
%disp(dER);

plot(difficulty, learning_rate);
%f3 = figure;
%plot(difficulty, error_rate.*difficulty);
%plot(error_rate, learning_rate);
%disp(learning_rate);
%disp(error_rate)
%[max, i] = max(learning_rate);
%best_error = error_rate(i);
%disp(best_error);


