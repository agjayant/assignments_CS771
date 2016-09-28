function [mu,z,score] = kmeans(X,K,init)
% input X is N*D data, K is number of clusters desired.  init is either
% 'random' which just chooses K random points to initialize with, or
% 'furthest', which chooses the first point randomly and then uses the
% "furthest point" heuristic.  the output is mu, which is K*D, the
% coordinates of the means, and z, which is N*1, and only integers 1...K
% specifying the cluster associated with each data point.  score is the
% score of the clustering
  
[N D] = size(X);

if K >= N,
  error('kmeans: you are trying to make too many clusters!');
end;

if nargin<3 || strcmp(init, 'random'),
  % initialize by choosing K (distinct!) random points: we do this by
  % randomly permuting the examples and then selecting the first K
  perm = randperm(N);
  perm = perm(1:K);
  
  % the centers are given by points
  mu = X(perm, :);
  
  % we leave the assignments unspecified -- they'll be computed in the
  % first iteration of K means
  z = zeros(N,1);

elseif strcmp(init, 'furthest'),
  % initialize the first center by choosing a point at random; then
  % iteratively choose furthest points as the remaining centers

  %TODO
  mu(1,:) = X(randperm(N,1),:);
  for i=2:K
      max = norm(mu(1,:) - X(1,:));
      mu(i,:) = X(1,:) 
      for j=1:length(mu)
          for k=1:N
              if max < norm(mu(j,:) - X(k,:));
                  max = norm(mu(j,:) - X(k,:));
                  mu(i,:) = X(k,:);
              end
          end
      end
  end
  
  % again, don't bother initializing z
  z = zeros(N,1);
else
  error('unknown initialization: use "furthest" or "random"');
end;

% begin the iterations.  we'll run for a maximum of 20, even though we
% know that things will *eventually* converge.
for iter=1:20,
  % in the first step, we do assignments: each point is assigned to the
  % closest center.  we'll judge convergence based on these assignments,
  % so we want to keep track of the previous assignment
  
  oldz = z;
  
  for n=1:N,
    % assign point n to the closest center
    %TODO
    z(n) = 1;
    min = norm(mu(1) - X(n,:));
    for i=2:length(mu)
        dist = norm(mu(i) - X(n,:));
        if min > dist
            min = dist;
            z(n) = i;
        end
    end
  end;
  
  % check to see if we've converged
  if all(oldz==z),
    break;  % break out of loop
  end;
  
  
  % re-estimate the means
  %TODO
  mu = zeros(K,D);
  for i=1:N
      mu(z(i),:) = mu(z(i),:) + X(i,:);
  end
  mu = mu/K;
  
  
end;

% final: compute the score
score = 0;
for n=1:N,
  % compute the distance between X(n,:) and it's associated mean
  score = score + norm(X(n,:) - mu(z(n),:))^2;
end;
