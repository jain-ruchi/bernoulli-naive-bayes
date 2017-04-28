function pred_probs = naive_bayes(train_data, train_label, eval_data)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  eval_data: M*D matrix, each row as a sample and each column as a
%
% Output:
%  pred_probs: prediction probabilities

classes = unique(train_label);
priors = zeros(rows(classes), 1);
estimates = zeros(rows(classes), columns(train_data));

for i = 1:rows(classes) % for each class
    class_i = train_data(train_label==classes(i),:); % extract data points belonging to that class
    estimates(i, :) = sum(class_i, 1) * (1/rows(class_i));
    priors(i, 1) = rows(class_i)/rows(train_data);
endfor

% set any zero elements to a very small probability
estimates(estimates==0) = 0.01;

pred_probs = zeros(rows(eval_data), rows(classes));

for k = 1:rows(eval_data)
    probs = estimates .^ eval_data(k,:);
    probs(probs==1) = 1 - estimates(probs==1);
    joint_probs = prod(transpose(probs));
    joint_probs = transpose(joint_probs) .* priors; % multiply by the priors
    pred_probs(k, :) = transpose(joint_probs);
endfor

return
endfunction