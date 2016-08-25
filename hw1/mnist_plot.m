load mnist_hw1.mat;
num = 40;
accuracy_vector = zeros([1 num]);
for i = 1:num
    accuracy_vector(i) = predict_digit(50*i,dataX,X_test,Y_test);
end

accuracy_vector = accuracy_vector*100;

X = 1:num;
X = X*50;

accuracy_all = ones([1 num]);
accuracy_all = accuracy_all*mnist_classifier(dataX,X_test,Y_test)*100;

figure1 = figure;
plot(X,accuracy_vector,'--*',X,accuracy_all);
title('MNIST CLASSIFIER');
xlabel('Number of Training Examples')
ylabel('Accuracy(%)')
legend('Using Random Samples from Training Examples','Using all Training Examples');
saveas(figure1,'plot_mnist.jpg');