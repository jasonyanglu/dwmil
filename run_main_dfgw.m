clear; clc; close all;

run_num = 1;

data_select = {'gaussian',...
                'sea',...
                'hyperP',...
                'checkerboard',...
                'electricity',...
                'weather'};


for data_i = 1:length(data_select)
    load(['data/' data_select{data_i}]);

for run_i = 1:run_num
    fprintf('run %d, data: %s\n', run_i, data_select{data_i});
    tic;
    
    dfgw_option.chunk_num = chunk_num;
    [ dfgw_pred_label] = dfgw( data, label, dfgw_option );
    result_dfgw(run_i) = chunk_measure( dfgw_pred_label, label, chunk_num );
    t_dfgw(run_i) = toc;
end

fprintf('auc: %f, gm: %f\n',mean([result_dfgw.auc]),mean([result_dfgw.gm]));

end
