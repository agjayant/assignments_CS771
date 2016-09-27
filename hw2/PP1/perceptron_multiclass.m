% You are only suppose to complete the "TODO" parts and the code should run
% if these parts are correctly implemented
clear all;close all;
load('mnist_big.mat');
[N D] = size(X_train);
Y_train = Y_train + 1;
Y_test = Y_test + 1;
num_classes = max(Y_train);
W = randn(D,num_classes);
b = randn(1,num_classes);
W_avg = W;
b_avg = b;
k = 0; % number of mistakes (i.e., number of updates)

% Just do a single pass over the training data (our stopping condition will
% simply be this)
for n=1:N
    % TODO: predict the label for the n-th training example
    % as y_pred = argmax W'x_n + b    
    [max_pred, y_pred] = max(transpose(transpose(W)*transpose(X_train(n,:))) + b);
    %[max_pred, y_pred] = max(transpose(transpose(W_avg)*transpose(X_train(n,:))) + b_avg);
    
    if y_pred ~= Y_train(n)
        k = k + 1;
        % TODO: Update W, b
        W(:,Y_train(n)) = W(:,Y_train(n)) + transpose(X_train(n,:));
        b(Y_train(n)) = b(Y_train(n)) + 1;
        W(:,y_pred) = W(:,y_pred) - transpose(X_train(n,:));
        b(y_pred) = b(y_pred) -1 ;
        
        % TODO: Update W_avg, b_avg using Ruppert Polyak Averaging
        % Important: You don't need to store all the previous W's and b's
        % to compute W_avg, b_avg
        W_avg = W_avg - (W_avg-W)/k;
        b_avg = b_avg - (b_avg-b)/k;
        
        % TODO: Predict test labels using W, b
        [T d] = size(X_test);
        for j=1:T
            [maxPred, y_test_pred(j)] = max(transpose(transpose(W)*transpose(X_test(j,:))) + b);
        %y_test_pred = % ?? TODO
        end
        acc(k) = mean(Y_test==transpose(y_test_pred));   % test accuracy     
        
        % TODO: Now predict test labels using W_avg, b_avg
        [T d] = size(X_test);
        for j=1:T
            [maxPred, y_test_pred(j)] = max(transpose(transpose(W_avg)*transpose(X_test(j,:))) + b_avg);
        %y_test_pred = % ?? TODO
        end
        acc_avg(k) = mean(Y_test==transpose(y_test_pred)); % test accuracy with R-P averaging
        
    end
end

fprintf('Update number %d, accuracy = %f, accuracy (with R-P averaging) = %f\n',k,acc(k),acc_avg(k));
plot(1:k,acc(1:k),'r');
hold on;
plot(1:k,acc_avg(1:k),'g');
drawnow;        