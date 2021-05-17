function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
deltas = h - y;

J = (1/(2*m)).*(deltas'*deltas);

temp = theta;  temp(1) = 0;
regularized_par = (lambda/(2*m)).*(temp'*temp);
J = J + regularized_par;

grad = (1/m).*(X'*deltas);
regularized_grad = (lambda/m).*temp;
grad = grad + regularized_grad;
% =========================================================================

grad = grad(:);

end
