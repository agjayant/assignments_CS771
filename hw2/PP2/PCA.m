function [Z,U,evals] = PCA(X,K)
  
% X is N*D input data, K is desired number of projection dimensions (assumed
% K<D).  Return values are the projected data Z, which should be N*K, U,
% the D*K projection matrix (the eigenvectors), and evals, which are the
% eigenvalues associated with the dimensions
  
[N D] = size(X);

if K > D,
  error('PCA: you are trying to *increase* the dimension!');
end;

% first, we have to center the data
%TODO
mean = zeros(1,D);
for i=1:N
    mean = mean + X(i,:);
end
mean = mean/N;

for i=1:N
    X(i,:)= X(i,:)-mean;
end

% next, compute the covariance matrix C of the data
%TODO

C = (transpose(X)*X)/N;

% compute the top K eigenvalues and eigenvectors of C... 
% hint: you may use 'eigs' built-in function of MATLAB

%TODO
[U V] = eig(C, 'vector');
[V ind]  = sort(V,'descend');   
U = U(:,ind(1:K));
evals = V(1:K);

% project the dataeval

Z = X*U;
