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

% next, compute the covariance matrix C of the data

%TODO

% compute the top K eigenvalues and eigenvectors of C... 
% hint: you may use 'eigs' built-in function of MATLAB

%TODO


% project the data

Z = X*U;
