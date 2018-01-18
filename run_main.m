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
    
    dwmil_options.chunk_num = chunk_num;
    dwmil_options.T = 11;
    dwmil_options.theta = 0.001;
    dwmil_options.err_type = 2;

    [ dwmil_pred_value ] = dwmil( data, label, dwmil_options );
    result_dwmil(run_i) = chunk_measure( dwmil_pred_value, label, chunk_num );
    t_dwmil(run_i) = toc;
end


fprintf('auc: %f, gm: %f\n',mean([result_dwmil.auc]),mean([result_dwmil.gm]));

end
