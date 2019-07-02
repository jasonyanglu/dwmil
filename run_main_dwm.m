clear; clc; close all;

addpath('IncrementalLearning-master/src/');
javaaddpath('weka.jar');
javaaddpath('moa.jar');

run_num = 1;

            
data_select = {'gaussian',...
                'sea',...
                'hyperP',...
                'checkerboard',...
                'electricity',...
                'weather'};

for data_i = 1:length(data_select)
    load(['data/' data_select{data_i}]);
    non_cell_data = [];
    non_cell_label = [];
    for chunk_i=1:chunk_num
        non_cell_data = [non_cell_data;data{chunk_i}];
        non_cell_label = [non_cell_label;label{chunk_i}];
    end
    
for run_i = 1:run_num
    fprintf('run %d, data: %s\n', run_i, data_select{data_i});
    tic;
    dwm_options.chunk_num = chunk_num;
    dwm_options.data_i = data_i;
    dwm_options.T = 11;
    [ dwm_pred_label] = dwm( non_cell_data, non_cell_label, dwm_options );
    data_num = length(dwm_pred_label);
    idx = 1;
    for chunk_i=1:chunk_num
        chunk_size = length(label{chunk_i});
        if idx+chunk_size <= data_num
            cell_pred_label{chunk_i} = dwm_pred_label(idx:idx+chunk_size-1);
        else
            cell_pred_label{chunk_i} = dwm_pred_label(idx:end);
        end
        idx = idx + chunk_size;
    end
    result_dwm(run_i) = chunk_measure( cell_pred_label, label, chunk_num );
    t_dwm(run_i) = toc;
    
end

fprintf('auc: %f, gm: %f\n',mean([result_dwm.auc]),mean([result_dwm.gm]));

end
