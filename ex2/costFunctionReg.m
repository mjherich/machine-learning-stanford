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

% % Start loop
% for i=1:m
%     x = X(i, :)'; % Define x
%     
%     % Parameterize Hypothesis Function
%     h = sigmoid( theta' * x );
%     
%     % Calculate Cost
%     J = J - y(i)*log(h) - (1 - y(i))*log(1-h);
%     
%     % Calculate Gradient
%     for j=1:length(theta)
%         if (j==1)
%             grad = grad + h - y(i);
%         else
%             grad = grad + ( h - y(i) ) * x(j);
%         end
%     end
%     
% end
% J = J/m; grad = grad/m;
% 
% % Initialize regularization temp variables
% JR = 0; gradR = 0;
% % Begin loop for regularization terms (indexed at 2 through length(theta))
% for i=2:length(theta)
%     % Calculate Cost Regularization
%     JR = JR + theta(i)^2;
%     
%     % Calculate Gradient Regularization
%     gradR = gradR + theta(i);
% end
% JR = lambda/(2*m) * JR;
% gradR = lambda/m * gradR;
% 
% % Put it all together
% J = J + JR;
% grad = grad + gradR;

h = sigmoid(X*theta);
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
shift_theta = theta(2:length(theta));
theta_reg = [0;shift_theta];

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

% grad_zero = (1/m)*X(:,1)'*(h-y);
% grad_rest = (1/m)*(shift_x'*(h - y)+lambda*shift_theta);
% grad      = cat(1, grad_zero, grad_rest);

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

% =============================================================

end
