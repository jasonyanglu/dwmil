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
    
    rea_option.chunk_num = chunk_num;
    [ rea_pred_label] = rea( data, label, rea_option );
    result_rea(run_i) = chunk_measure( rea_pred_label, label, chunk_num );
    t_rea(run_i) = toc;
end



fprintf('auc: %f, gm: %f\n',mean([result_rea.auc]),mean([result_rea.gm]));

end
