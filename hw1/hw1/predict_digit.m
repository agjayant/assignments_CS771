function accuracy = predict_digit(T,dataX,X_test, Y_test)

mean_all= zeros([10 784]);
for i = 1:10
	[N,D] = size(dataX{i});
    sample_list = randperm(N);
	for j = sample_list(1:T)
		mean_all(i,:) = mean_all(i,:) + dataX{i}(j,:); 
	end
	mean_all(i,:) = mean_all(i,:)/T;
end

[N,D] = size(X_test);
predict = zeros([1 N]);

for i = 1:N
	min_norm = norm(X_test(i,:)-mean_all(1,:)) ;
	predict(i) = 0;
	for j = 2:10
		temp = norm(X_test(i,:) - mean_all(j,:));
		if temp < min_norm
			min_norm = temp;
			predict(i) = j-1;
		end
	end
end
	
correct_pred = 0;

for i= 1:N
	if predict(i) == Y_test(i)
		correct_pred = correct_pred + 1;
	end
end

accuracy = correct_pred/N;

end


