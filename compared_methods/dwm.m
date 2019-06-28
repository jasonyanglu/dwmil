function [ pred_label ] = dwm( data, label, options )
%DWM Summary of this function goes here
%   Detailed explanation goes here

    [data_num, fea_num] = size(data);
    label_value = unique(label);
    pred_label = zeros(data_num,1);
    theta = 0.9;
    th = 0.001;
    ensemble = [];
    r = [0.5 0.5];
    T = options.T;
    
    

%     moa
%     convert to arff file format
    fea_name = {'1','2','label'};
    wekaOBJ = matlab2weka_Instances('training', fea_name, data, label, 3);
    saveARFF([num2str(options.data_i) 'temp.arff'],wekaOBJ);
    
    % load arff file
%     fs = moa.streams.ArffFileStream([num2str(options.data_i) '.arff'],3);
    fs = moa.streams.ArffFileStream([num2str(options.data_i) 'temp.arff'],3);

    % initialize classifier ensembles
    import moa.classifiers.trees.HoeffdingTree.*; 

    crt_data = fs.nextInstance();
    
    m = 1;
    w = 1;
    p = round(data_num / options.chunk_num);
    
    for t=1:T
        ensemble.train_model{m}{t} = moa.classifiers.trees.HoeffdingTree();
        ensemble.train_model{m}{t}.setModelContext(fs.getHeader());
        ensemble.train_model{m}{t}.prepareForUse();
        ensemble.train_model{m}{t}.trainOnInstance(crt_data);
    end
    

    for data_i=2:data_num

        crt_data = fs.nextInstance();

        % predict
        pred_label_single = [];
        for i=1:m
            for t=1:T
                pred_prob = ensemble.train_model{i}{t}.getVotesForInstance(crt_data);
                [~,pred_label_single(i,t)] = max(pred_prob);
                pred_label_single(i,t) = label_value(pred_label_single(i,t));
            end
            pred_label_single = sign(mean(pred_label_single,2));
            if pred_label_single(i) ~= label(data_i) && mod(data_i,p)==0
                w(i) = w(i) * 0.5;
            end
        end
        pred_label(data_i) = sign(w * pred_label_single);

        if label(data_i) == 1
            r(1) = theta * r(1) + (1-theta);
            r(2) = theta * r(2);
        else
            r(1) = theta * r(1);
            r(2) = theta * r(2) + (1-theta);
        end
        
        if mod(data_i,p)==0
            w = w / sum(w);
            remove_idx = w < th;
            ensemble.train_model(remove_idx) = [];
            w(remove_idx) = [];
            m = m - sum(remove_idx);
            if pred_label(data_i) ~= label(data_i)
                m = m + 1;
                w(m) = 1;
                for t=1:T
                    ensemble.train_model{m}{t} = moa.classifiers.trees.HoeffdingTree();
                    ensemble.train_model{m}{t}.setModelContext(fs.getHeader());
                    ensemble.train_model{m}{t}.prepareForUse();
                    ensemble.train_model{m}{t}.trainOnInstance(crt_data);
                end
            end
        end
        
        % increamental train 
        for i=1:m
            for t=1:T
                if (label(data_i) == -1 && r(1)/r(2) > rand) || label(data_i) == 1
                    ensemble.train_model{i}{t}.trainOnInstance(crt_data);
                end
            end
        end
    end

end
