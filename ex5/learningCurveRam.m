
function [error_train, error_val] = ...
    learningCurveRam(X, y, Xval, yval, lambda)

    % you need these values later
    m = size(X,1)  ;   % the number of training examples
    r = size(Xval,1);  % the number of validation examples

    %for i = 1:m
      % create two empty vectors for the Jtrain and Jcv values
      %for j = 1:50
        % use 'm' to select 'i' random examples from the training set
        % use 'r' to select 'i' random examples from the validation set
        % compute theta
        % compute Jtrain and Jcv and save the values
     % end
      % compute the mean of the Jtrain vector and save it in error_train(i)
      % compute the mean of the Jcv vector and save it in error_val(i)
    %end

    error_train = zeros(m, 1);
    error_val   = zeros(m, 1);

    for i = 1:m
        Jtrain = zeros(50, 1);
        Jcv   = zeros(50, 1);

        for j = 1:50
            
            k = randperm(i);
            kk = randperm(r);
            
            theta = trainLinearReg(X(k,:), y(k), lambda);
            Jtrain(i) = linearRegCostFunction(X(k,:), y(k), theta, 0);
            Jcv(i) = linearRegCostFunction(Xval(kk,:), yval(kk), theta, 0);
        end

        error_train(i) = mean(Jtrain);
        error_val(i) = mean(Jcv);
    end

end

