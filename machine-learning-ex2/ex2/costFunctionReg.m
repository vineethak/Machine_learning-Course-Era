function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

ntheta1 = theta(2:end);
ntheta = [0;ntheta1;];
h_thetaX = sigmoid(X*theta);
J = (1/m)*(-y'*log(h_thetaX) - (1 - y)'*log(1. - h_thetaX)) + (lambda/(2*m))*(ntheta'*ntheta);
%ntheta(1) = 0;
grad =   (1/m)*((X'*(h_thetaX - y)) + (lambda*ntheta));

% =============================================================

end
