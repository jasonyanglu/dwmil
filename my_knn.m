function [ nn_idx ] = my_knn( sel_data, data, catidx, k)
%MY_KNN Summary of this function goes here
%   Detailed explanation goes here

[sel_num, fea_num] = size(sel_data);
k = min(k,sel_num-1);
data_num = size(data,1);

cat_num = length(catidx);
num_idx = setdiff(1:fea_num,catidx);
med_std = median(std(data(:,num_idx)));

cat_mat = zeros(sel_num,data_num);
for i=1:sel_num
    for j=1:data_num
        for l=1:cat_num
            cat_mat(i,j) = cat_mat(i,j) + double(sel_data(i,catidx(l))~=data(j,catidx(l)));
        end
    end
end

num_pd = pdist2(sel_data(:,num_idx), data(:,num_idx));
sq_dist = sqrt(num_pd.^2 + med_std^2 * cat_mat);

[~,nn_idx] = sort(sq_dist,2,'ascend');
nn_idx = nn_idx(:,2:k+1)';
    
end

