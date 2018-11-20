function graphFig1()
% This function produces four figures for Fig. 1 of The Eighty Five Percent
% Rule for Optimal Learning (not exactly the same, but shows that they are
% obtainable).
%
%
% Author: Vannesa Nguyen              Last Update: November 19, 2018

% Initializations
increment = 0.1;
max_diff = 100;
difficulty = [0: increment: max_diff];
k = length(difficulty);

% Variables for before learning curves
before_delta = 30;
error_rate2 = zeros(k,1);
learning_rate2 = zeros(k,1);

% Variables for after learning curves
after_delta = 15;
error_rate = zeros(k,1);
learning_rate = zeros(k,1);


for diff_i = 1:k
    % Create y values for before learning curves
    gaussian_curve2  = (normcdf([-inf, 0], difficulty(diff_i), before_delta));
    error_rate2(diff_i,:) = gaussian_curve2(2);
    pdf_curve2  = (normpdf([-inf, 0], difficulty(diff_i), before_delta));
    learning_rate2(diff_i,:) = (pdf_curve2(2) - pdf_curve2(1)) * difficulty(diff_i);
    
    % Create y values for after learning curves
    gaussian_curve  = (normcdf([-inf, 0], difficulty(diff_i), after_delta));
    error_rate(diff_i,:) = gaussian_curve(2);
    pdf_curve  = (normpdf([-inf, 0], difficulty(diff_i), after_delta));
    learning_rate(diff_i,:) = (pdf_curve(2) - pdf_curve(1)) * difficulty(diff_i);    
end


% Fig 1A
h = [-100: increment : 125];
f0 = figure;
plot(h, normpdf(h, 16, after_delta), h, normpdf(h, 16, before_delta));
title('Fig. 1A of paper; distribution of decision variable h')
xlabel('Decision variable, h')
ylabel('Probability of h');

% Fig 1B
f1 = figure;
plot(difficulty, error_rate, difficulty, error_rate2);
title('Fig. 1B of paper; error rate as a function of difficulty')
xlabel('Decision variable, h')
ylabel('Error rate, ER');

% Fig 1C
f2 = figure;
plot(difficulty, learning_rate./2, difficulty, learning_rate2);
title('Fig. 1C of paper; learning rate as a function of difficulty')
xlabel('Decision variable, h')
ylabel('Learning rate, dER/dB');

% Fig 1D
f3 = figure;
plot(error_rate, learning_rate./2, error_rate2, learning_rate2);
title('Fig. 1D of paper; learning rate as a function of error rate')
xlabel('Error rate, ER')
ylabel('Learning rate, dER/dB');


end


