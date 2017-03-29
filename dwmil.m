function [ final_pred ] = dwmil( data, label, options )
%DWMDS Summary of this function goes here
%   Detailed explanation goes here

    chunk_num = options.chunk_num;
    model = [];
    final_pred{1} = zeros(length(label{1}),1);
    m = 0;
    
    for chunk_i = 1:chunk_num
        
        crt_data = data{chunk_i};
        crt_label = label{chunk_i};
        pos_idx = crt_label==1;
        neg_idx = crt_label==-1;
        pos_num = sum(pos_idx);
        neg_num = sum(neg_idx);

        % create new base classifier
        m = m + 1;
        model{m} = sampling_bagging( crt_data, crt_label, options );
        w(m,1) = 1;
        
        if chunk_i > 1
            % get prediction from current ensemble
            all_pred = predict_base(crt_data,model);
            pred_value = all_pred(:,1:end-1) * (w(1:end-1));
            final_pred{chunk_i} = pred_value;
            pred = sign(pred_value);
            pred(pred==0) = 1;

            % error for each base classifier
            err = calculate_err(all_pred, crt_label, options.err_type);
            
            % update weight and remove classifiers
            w = (1 - err) .* w;
            
            remove_idx = w < options.theta;
            model(remove_idx) = [];
            w(remove_idx) = [];
            m = m - sum(remove_idx);
            
        end
    end
end

    function model = sampling_bagging( data, label, options )
        

        T = options.T;
        
        os_rate = 0;
        us_rate = 1;
        

        pos_idx = label==1;
        neg_idx = label==-1;
        pos_num = sum(pos_idx);
        neg_num = sum(neg_idx);
        
        
        model = cell(T,1);
        if neg_num > pos_num
            sampling_num_pos = round(pos_num + (neg_num - pos_num) * os_rate);
            sampling_num_neg = round(neg_num - (neg_num - pos_num) * us_rate);
        else
            sampling_num_pos = round(pos_num - (pos_num - neg_num) * us_rate);
            sampling_num_neg = round(neg_num + (pos_num - neg_num) * os_rate);
        end
        for t=1:T
            
            
            sampling_data_pos = datasample(data(pos_idx,:), sampling_num_pos);
            sampling_data_neg = datasample(data(neg_idx,:), sampling_num_neg);

            sampling_data = [sampling_data_pos; sampling_data_neg];
            sampling_label = [ones(sampling_num_pos,1); -ones(sampling_num_neg,1)];

            data_num = size(sampling_data,1);
            rand_idx = randperm(data_num);
            sampling_data = sampling_data(rand_idx,:);
            sampling_label = sampling_label(rand_idx);
            
            model{t} = fitctree(sampling_data,sampling_label);
            
        end

    end


    function err = calculate_err( all_pred, label, error_type)
        
        m = size(all_pred,2);
        err = zeros(m,1);
        
        for i=1:m
            
            tp=sum(label==1 & all_pred(:,i)==1);
            fn=sum(label==1 & all_pred(:,i)==-1);
            tn=sum(label==-1 & all_pred(:,i)==-1);
            fp=sum(label==-1 & all_pred(:,i)==1);
            
            if(tp==0)    
                f1=0;
                gm=0;
            else
                prec=tp/(tp+fp);
                rec_pos=tp/(tp+fn);
                rec_neg=tn/(tn+fp);
                f1=2*(prec*rec_pos)/(prec+rec_pos);
                gm=sqrt(rec_pos*rec_neg);
                wrm = 0.5*(1-rec_pos) + 0.5*(1-rec_neg);
            end

            if error_type == 1
                err(i) = 1-f1;
            elseif error_type == 2
                err(i) = 1-gm;
            elseif error_type == 3
                err(i) = wrm;
            end
            
        end
            
    end


    function pred = predict_base(data,model)
        
        m = length(model);
        T = length(model{1});
        for i=1:m
            for t=1:T
                pred_single(:,t) = predict(model{i}{t},data);
            end
            pred(:,i) = sign(sum(pred_single,2));
        end

    end