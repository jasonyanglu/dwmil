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
    
    lpn_options.chunk_num = chunk_num;
    lpn_options.a = 0.5;
    lpn_options.b = 10;
    lpn_options.err_type = 2;

    [ lpn_pred_value ] = lpn( data, label, lpn_options );
    result_lpn(run_i) = chunk_measure( lpn_pred_value, label, chunk_num );
    t_lpn(run_i) = toc;
    
end

fprintf('auc: %f, gm: %f\n',mean([result_lpn.auc]),mean([result_lpn.gm]));


end
