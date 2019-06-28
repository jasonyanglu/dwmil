function [ final_pred] = lpn( data, label, options )

%LPN Summary of this function goes here
%   Detailed explanation goes here

    chunk_num = options.chunk_num;
    model = [];
    final_pred{1} = zeros(length(label{1}),1);
    
    beta = zeros(chunk_num,chunk_num);
    
    for chunk_i = 1:chunk_num
        
        crt_data = data{chunk_i};
        crt_label = label{chunk_i};
        min_idx = crt_label==1;
        maj_idx = crt_label==-1;
        min_num = sum(min_idx);
        maj_num = sum(maj_idx);
        
        
        model{chunk_i} = bagging_variation(crt_data, crt_label);
        all_pred = predict_base(crt_data,model); % n-by-m
        
        
        if chunk_i > 1
            pred_value = all_pred(:,1:end-1) * w;
            final_pred{chunk_i} = pred_value;
        end
        
        err = calculate_err( all_pred, crt_label, options.err_type);
        
        if err(chunk_i) > 0.5
            model{chunk_i} = bagging_variation(crt_data, crt_label);
            all_pred(:,chunk_i) = predict_base(crt_data,model(chunk_i)); % n-by-m
            err(chunk_i) = calculate_err( all_pred(:,chunk_i), crt_label, options.err_type);
            if err(chunk_i) > 0.5
                err(chunk_i) = 0.5;
            end
        end
        
        err(err>0.5) = 0.5;
        
        beta(1:chunk_i,chunk_i) = err ./ (1 - err);
        
        
        for k=1:chunk_i
            omega = 1:(chunk_i - k + 1); 
            omega = 1./(1+exp(-options.a*(omega-options.b)));
            omega = (omega/sum(omega))';
            beta_hat = sum(omega.*(beta(k,k:chunk_i)'));
            w(k,1) = log(1 ./ beta_hat);
        end
        
             
        
        
    end

end

    function model = bagging_variation( data, label )
        
        T = 5;
        
        pos_idx = label==1;
        neg_idx = label==-1;
        pos_num = sum(pos_idx);
        neg_num = sum(neg_idx);
        
        model = cell(T,1);
        if pos_num < neg_num
            sampling_num_pos = pos_num;
            sampling_num_neg = round(size(data,1) / T);
        else
            sampling_num_pos = round(size(data,1) / T);
            sampling_num_neg = neg_num;
        end
        for t=1:11
            
            sampling_data_pos = datasample(data(pos_idx,:), sampling_num_pos, 'Replace', false);
            sampling_data_neg = datasample(data(neg_idx,:), sampling_num_neg, 'Replace', false);
            
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