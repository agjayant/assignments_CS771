X = load('test_pca.txt');


figure(1);
plot(X(:,1),X(:,2),'bo');
mypause

[Z,vecs,vals] = PCA(X,2);

XX = Z(:,1)*pinv(Z(:,1))*(X-ones(100,1)*mean(X))+ones(100,1)*mean(X);
figure(2);
plot(X(:,1),X(:,2),'bo',XX(:,1),XX(:,2),'ro');
hold on;
for n=1:100,
  plot([X(n,1) XX(n,1)],[X(n,2) XX(n,2)], 'k-');
end;
hold off;

XX = Z(:,2)*pinv(Z(:,2))*(X-ones(100,1)*mean(X))+ones(100,1)*mean(X);
figure(3);
plot(X(:,1),X(:,2),'bo',XX(:,1),XX(:,2),'ro');
hold on;
for n=1:100,
  plot([X(n,1) XX(n,1)],[X(n,2) XX(n,2)], 'k-');
end;
hold off;
