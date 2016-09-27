close all;
% % first, load the results if they exist
% h = fopen('results.mat','rb');  % open to check if the file exists
% if h >= 0,  % open successful -- read data
%   fclose(h);
%   fprintf('loading data from results.mat\n');
%   load results;
% else  % otherwise, start from beginning
%   fprintf('initializing empty results\n');
%   
% end;

fname = 'mnist';
fprintf('loading data from %s.mat\n', fname);
load(fname);
results = {};


% rename variables to make life easier
X = trainX;
Y = trainY;

fprintf(['\n\nIn our first experiment, we will cluster the digits into K\n' ...
         'clusters, for K varying from 2 to 20, looking at scores for\n' ...
         'each result.  We will run 3 times for each K and choose the\n' ...
         'best score.\n\n']);


allK = [2 5 10 15 20];

if ~isfield(results, 'kmeansscores'),
  tic;
  scores = zeros(size(allK));
  zs     = {};
  mus    = {};
  for ii=1:length(allK),
    fprintf('  K=%d...', allK(ii));
    bestScore = Inf;
    bestZ     = 0;
    bestMu    = 0;
    for rep=1:3,
      [mu,z,s] = kmeans(X,allK(ii),'furthest');
      if s < bestScore,
        bestScore = s;
        bestZ = z;
        bestMu = mu;
      end;
      fprintf('.'); 
    end;
    fprintf(' --> score %g\n', bestScore); 
    
    scores(ii) = bestScore;
    zs{ii} = bestZ;
    mus{ii} = bestMu;
  end;
  
  results.kmeansscores = scores;
  results.kmeanszs     = zs;
  results.kmeansmus    = mus;

  results.kmeansaccs = zeros(length(allK), 4);
  for ii=1:length(allK),
    [acc,ri,p,r,f,nes,vi,nvi] = cluster_eval(Y, results.kmeanszs{ii});
    results.kmeansaccs(ii,:) = [ri f nes nvi];
  
    N = length(Y);
    results.kmeansbic(ii) = N * log(results.kmeansscores(ii) / N) + allK(ii) * log(N);
    results.kmeansaic(ii) = N * (log(2*pi*results.kmeansscores(ii) / N) + 1) + 2 * allK(ii);
  end;
  results.kmeansscores = results.kmeansscores / max(results.kmeansscores);
  results.kmeansbic = results.kmeansbic / max(results.kmeansbic);
  results.kmeansaic = results.kmeansaic / max(results.kmeansaic);
  
  toc;
end;
figure(1);

plot(allK, results.kmeansscores, 'bx-', ...
   allK, results.kmeansbic   , 'ro-', ...
   allK, results.kmeansaic   , 'k+-');
legend('normalized score', 'normalized BIC','normalized AIC','LOCATION','SouthWest');
  
xlabel('K');
ylabel('score');
title('fig 1: K versus score for digits data');

figure(2);

plot(allK, results.kmeansaccs(:,1), 'bx-', ...
   allK, results.kmeansaccs(:,2), 'ro-', ...
   allK, results.kmeansaccs(:,3), 'k+-', ...
   allK, results.kmeansaccs(:,4), 'ms-');
legend('rand index', 'fscore', 'edit score', 'normalized information','LOCATION','SouthEast');

xlabel('K');
ylabel('score');
title('fig 2: K versus "classification" accuracy');

figure(3); draw_digits(results.kmeansmus{2}, 1:allK(2));
figure(4); draw_digits(results.kmeansmus{3}, 1:allK(3));
figure(5); draw_digits(results.kmeansmus{4}, 1:allK(4));

mypause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(['Next, we run PCA and project down to three dimensions.  We will\n' ...
         'plot all pairs of dimensions so we can get a sense of how things\n' ...
         'break down.\n\n']);

mypause;

if ~isfield(results, 'pcavecs'),
  tic
  [Z,vecs,vals] = PCA(X,50);   % for now, do ten dimensions just so we get the eigenvalues
  results.pcaZ = Z;
  results.pcavecs = vecs;
  results.pcavals = vals;
  toc;
end;


figure(6);
plot(1:50, results.pcavals, 'bx-');
title('fig 6: PCA eigenvalues');
xlabel('dimensionality');
ylabel('eigenvalue');

figure(7);
Z = results.pcaZ;
colors = ['bx';'rx';'kx';'mx';'gx';'cx';'yx'];
subplot(1,3,1);
plot(Z(:,1), Z(:,2), 'w.');
hold on;
for y=1:5,
  plot(Z(Y==y,1), Z(Y==y,2), colors(y,:));
end;
hold off;
title('blue=1, red=2');

subplot(1,3,2);
plot(Z(:,3), Z(:,2), 'w.');
hold on;
for y=1:5,
  plot(Z(Y==y,3), Z(Y==y,2), colors(y,:));
end;
hold off;
title('black=3');

subplot(1,3,3);
plot(Z(:,1), Z(:,3), 'w.');
hold on;
for y=1:5,
  plot(Z(Y==y,1), Z(Y==y,3), colors(y,:));
end;
hold off;
title('magenta=4, green=5');

mypause


fprintf(['Now, we look at restorations of the original images based on the\n' ...
         'eigen-decomposition.\n\n']);



allD = [3 6 12 25 50];
for ii=1:length(allD),
  N = size(X,1);
  XX = Z(:,1:allD(ii))*(pinv(Z(:,1:allD(ii)))*(X-ones(N,1)*mean(X)))+ones(N,1)*mean(X);
  err = sum(sum((X-XX).*(X-XX))) / prod(size(X));
  figure(7+ii);
  %title(sprintf('fig %d: restored images with %d dims'));
  draw_digits(XX(1:15:end,:), Y(1:15:end));
  fprintf('  dim %d --> sum eig %g --> restoration error %g\n', allD(ii), sum(results.pcavals(1:allD(ii))), err);
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(['\nNow, we test to see how clustering and dimensionality reduction\n' ...
         'affect *classification* performance with 1NN.\n\n']);


if ~isfield(results, 'knnbaseline'),
  tic
  results.knnbaseline = TrainTest(@KNN, trainX, trainY, testX, testY, 1);
  results.knnbaselinetime = toc;
end;

fprintf('\nBaseline 1nn accuracy is %g\n', results.knnbaseline);


allK = [1 2 4 8 16];
if ~isfield(results, 'knncluster'),
  tic;
  results.knncluster = zeros(size(allK));
  for ii=1:length(allK),
    % compute k clusters for each class, and then take the mean of each cluster
    % as a "representative" point
    Xcl = [];
    Ycl = [];
    for y=0:9,
      mu = kmeans(X(Y==y,:),allK(ii),'furthest');
      Xcl = [Xcl ; mu];
      Ycl = [Ycl ; y*ones(size(mu,1),1)];
    end;
    fprintf('  %d clusters... ', allK(ii)); 
    tic
    results.knncluster(ii) = TrainTest(@KNN, Xcl, Ycl, testX, testY, 1);
    results.knnclustertime(ii) = toc;
  end;
  toc
end;

figure(13);
subplot(2,1,1);
title('fig 13a: knn with clustered training data');
plot([1 16], [results.knnbaseline results.knnbaseline], 'k-', ...
   allK, results.knncluster, 'bo-');
legend('no clustering', 'clustering','LOCATION','SouthEast');
xlabel('number of clusters');
ylabel('accuracy');

subplot(2,1,2);
title('fig 13b: knn with clustered training data');
plot([1 16], [results.knnbaselinetime results.knnbaselinetime], 'k-', ...
   allK, results.knnclustertime, 'bo-');
legend('no clustering', 'clustering','LOCATION','SouthEast');
xlabel('number of clusters');
ylabel('test time');

allD = [3 6 12 25 50];
if ~isfield(results, 'knnpca'),
  tic;
  results.knnpca = zeros(size(allD));
  for ii=1:length(allD),
    N = size(X,1); Z = (X-ones(N,1)*mean(X))*results.pcavecs(:,1:allD(ii));
    trX = Z*(pinv(Z)*(X-ones(N,1)*mean(X)))+ones(N,1)*mean(X);
    N = size(testX,1);  Z = (testX-ones(N,1)*mean(X))*results.pcavecs(:,1:allD(ii));
    teX = Z*(pinv(Z)*(testX-ones(N,1)*mean(X)))+ones(N,1)*mean(X);
    tic;
    fprintf('  %d dimensions... ', allD(ii)); 
    results.knnpca(ii) = TrainTest(@KNN, trX, Y, teX, testY, 1);
    results.knnpcatime(ii) = toc;
  end;
  toc;
end;
  
figure(14);
subplot(2,1,1);
title('fig 14a: knn with dimensionality reduction');
plot([3 50], [results.knnbaseline results.knnbaseline], 'k-', ...
   allD, results.knnpca, 'bo-');
legend('no reduction', 'pca','LOCATION','SouthEast');
xlabel('number of dimensions');
ylabel('accuracy');

subplot(2,1,2);
title('fig 14b: knn with dimensionality reduction');
plot([3 50], [results.knnbaselinetime results.knnbaselinetime], 'k-', ...
   allD, results.knnpcatime, 'bo-');
legend('no reduction', 'pca','LOCATION','SouthEast');
xlabel('number of dimensions');
ylabel('test time');

