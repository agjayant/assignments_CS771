close all;clear;

load('movielens100k.mat');
% X: matrix containing 100k ratings. Each rating is between 1-5 (zeros in X will be ignored)
% train_mask: binary mask matrix denotes which locations of X are "observed" (80k ratings)
% test_mask:  binary mask matrix denotes which locations of X have to be predicted (20k ratings)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = 5; % number of user/movie latent factors. keep it fixed  (should do reasonably)
[N M] = size(X);

% randomly initialize the user and movie latent factor matrices
U = randn(N,K);
V = randn(M,K);

lambda = 10; % l2 regularization constant for each latent factor (same for U and V). keep it fixed (should do reasonably)
num_iters = 100; % number of iteratation of the alternating optimization procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter=1:num_iters
    % TODO: update U, one row at a time
    for n=1:N        
        first = 0;
        second = zeros(1,K);
        mask = find(train_mask(n,:));
        for j=1:size(mask,2)
            first = first + transpose(V(mask(j),:))*V(mask(j),:);
            second = second + X(n,mask(j))*V(mask(j),:);
        end 
        first = first + lambda*eye(K);  
        
        U(n,:) = inv(first)*transpose(second);
    end
    
    % TODO: update V, one row at a time
    for m=1:M
        first = 0;
        second = zeros(1,K);
        mask=find(train_mask(:,m));
        for j=1:size(mask,1)
            first = first + transpose(U(mask(j),:))*U(mask(j),:);
            second = second + X(mask(j),m)*U(mask(j),:);
        end 
        first = first + lambda*eye(K);        
        
        V(m,:) = inv(first)*transpose(second);
    end

    % Mean absolute error (MAE) on training and test entries of X
    train_mae(iter) = sum(abs(nonzeros((X - U*V').*train_mask)))/nnz(train_mask);
    test_mae(iter) = sum(abs(nonzeros((X - U*V').*test_mask)))/nnz(test_mask);
    
    fprintf('Train MAE = %f, Test MAE = %f\n',train_mae(iter),test_mae(iter));
    plot(train_mae(1:iter),'r');
    hold on;
    plot(test_mae(1:iter),'g');
    legend('Train MAE','Test MAE')
    drawnow;
end