function [ final_pred ] = rea( data, label, options )
%REA Summary of this function goes here
%   Detailed explanation goes here


    chunk_num = options.chunk_num;
    model = [];
    final_pred{1} = zeros(length(label{1}),1);
    
    
    f = 0.5;
    k = 10;
    G_data = [];
    
    
    for chunk_i = 1:chunk_num
        
        crt_data = data{chunk_i};
        crt_label = label{chunk_i};
        crt_chunk_size = length(crt_label);
        
        
        pos_idx = crt_label==1;
        neg_idx = crt_label==-1;
        find_pos_idx = find(crt_label==1);
        pos_num = sum(pos_idx);
        neg_num = sum(neg_idx);
        gamma = pos_num / neg_num;
        
        
        if chunk_i > 1
            
            all_pred = predict_base(crt_data,model); % n-by-m
            pred_value = all_pred * w';
            final_pred{chunk_i} = pred_value;
        
        end
        
        rand_idx = randperm(crt_chunk_size);
        crt_data = crt_data(rand_idx,:);
        crt_label = crt_label(rand_idx);
        
        if f > (chunk_i-1) * gamma && chunk_i > 1
            crt_data_p = [crt_data;G_data];
            crt_label_p = [crt_label;ones(size(G_data,1),1)];
        elseif chunk_i == 1
            crt_data_p = crt_data;
            crt_label_p = crt_label;
        else
            knn_idx = my_knn( G_data, crt_data, [], k);
            delta = sum(ismember(knn_idx,find_pos_idx));
            [~,sort_idx] = sort(delta,'descend');
            add_idx = sort_idx(1:min(round((f - gamma) * crt_chunk_size),length(sort_idx)));
            crt_data_p = [crt_data;G_data(add_idx,:)];
            crt_label_p = [crt_label;ones(length(add_idx),1)];
        end
        
        model{chunk_i} = fitctree(crt_data_p,crt_label_p);
        pred = predict_base(crt_data,model);
        err = mean(pred ~= repmat(crt_label,[1,chunk_i]),1);
        w = log(1./err);
        
        G_data = [G_data;crt_data(pos_idx,:)];
        
    end

end



    function pred = predict_base(data,model)


        T = length(model);

        for t=1:T
            pred(:,t) = predict(model{t},data);
        end


    end