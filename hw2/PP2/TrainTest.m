function [acc,Y,model,trainTime,testTime] = TrainTest(learn, trainX, trainY, testX, testY, varargin)
  % usage: TrainTest(learn, trainX, trainY, testX, testY, ...)
  %   learn is a learning algorithm
  %   trainX/trainY is training data
  %   testX is test data
  %   testY is test labels (can be empty if unknown)
  %   remaining arguments are extra arguments to learning algorithm
    
  if nargin < 6,
    error;
  end;
  
  tic;
  model = feval(learn, 'train'  , trainX, trainY, varargin{:});
  trainTime = toc;

  tic;
  Y     = feval(learn, 'predict', model,  testX);
  testTime = toc;

  acc = 0;
  if ~isempty(testY),
    acc = mean(Y==testY);
    fprintf(1, 'accuracy %g%%\n', acc*100);
  end;
  
  