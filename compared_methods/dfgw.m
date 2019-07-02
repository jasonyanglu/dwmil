function [ final_pred ] = dfgw( data, label, options )
%DFGW Summary of this function goes here
%   Detailed explanation goes here


    chunk_num = options.chunk_num;
    model = [];
    final_pred{1} = zeros(length(label{1}),1);
    s = 1;
    
    fea_num = size(data{1},2);
    
    l = 1;
    for i=1:fea_num-1
        nck = nchoosek(1:fea_num,i);
        for j=1:size(nck,1)
            fea_comb{l} = nck(j,:);
            l = l + 1;
        end
    end
        
    fea_subspace_num = min(length(fea_comb), 49);
            
    fea_subspace_rand_idx = randperm(length(fea_comb));
    fea_idx = fea_comb(fea_subspace_rand_idx(1:fea_subspace_num));
    fea_subspace_num = fea_subspace_num + 1;
    fea_idx{fea_subspace_num} = 1:fea_num;
    
    
    
    for chunk_i = 1:chunk_num
        
        crt_data = data{chunk_i};
        crt_label = label{chunk_i};
        pos_idx = crt_label==1;
        neg_idx = crt_label==-1;
        pos_num = sum(pos_idx);
        neg_num = sum(neg_idx);
        
        
        if chunk_i > 1
            previous_data = data{chunk_i-1};
            final_pred{chunk_i} = test_dfgw(crt_data, previous_data, model, fea_idx, wd); % n-by-m
        end
        
        delta = neg_num;
        [model, wd, s] = train_dfgw( chunk_i, data(1:chunk_i), label(1:chunk_i), s, delta, fea_idx );
        
    end

end

    

    function [model, wd, s] = train_dfgw(t, data, label, s, delta, fea_idx)
    
        pos_idx = label{t}==1;
        neg_idx = label{t}==-1;
        pos_num = sum(pos_idx);
        
        P_data_size = 0;
        for i=s:t-1
            P_data_size = P_data_size + sum(label{i}==1);
        end
        
        if pos_num + P_data_size > delta
            s = s + 1;
        end
        P_data = [];
        ts = [];
        for i=s:t
            P_data = [P_data;data{i}(label{i}==1,:)];
            ts = [ts;i*ones(sum(label{i}==1),1)];
        end
        N_data = data{t}(neg_idx,:);
        
        pos_num = size(P_data,1);
        neg_num = size(N_data,1);
        rand_idx = randperm(pos_num);
        P_data = P_data(rand_idx,:);
        ts = ts(rand_idx);
        N_data = N_data(randperm(neg_num),:);
        
        pos_train_num = round(0.85 * size(P_data,1));
        neg_train_num = round(0.85 * size(N_data,1));
        train_data = [P_data(1:pos_train_num,:);N_data(1:neg_train_num,:)];
        train_label = [ones(pos_train_num,1);-ones(neg_train_num,1)];
        train_ts = ts(1:pos_train_num);
        hold_data = [P_data(pos_train_num+1:end,:);N_data(neg_train_num+1:end,:)];
        hold_label = [ones(pos_num-pos_train_num,1);-ones(neg_num-neg_train_num,1)];
        
        model = [];
        
        fea_set_num = length(fea_idx);
        for fea_i = 1:fea_set_num
            model{fea_i} = LearnH(train_data(:,fea_idx{fea_i}),train_label,11,train_ts);
            hold_pred(:,fea_i) = predict_base(hold_data(:,fea_idx{fea_i}),model{fea_i});
        end
        
        
        c = ones(length(hold_label),1);
        c(hold_label == 1) = sum(hold_label==-1) / sum(hold_label==1);
        
        Aeq = ones(1,fea_set_num);
        beq = 1;
        lb = zeros(fea_set_num,1);
        ub = ones(fea_set_num,1);
        x0 = ones(fea_set_num,1) / fea_set_num;
        fun = @(w) c' * log(1+exp(-hold_label .* (hold_pred * w)));
        opt_options = optimoptions('fmincon','Display','off');
        wd = fmincon(fun,x0,[],[],Aeq,beq,lb,ub,[],opt_options);
        
        
        
    end

    function pred = test_dfgw(data, previous_data, model, fea_idx, wd)
    
        bin_num = 30;
        
        for i=1:size(data,2)
            bin_min = min([data(:,i);previous_data(:,i)]);
            bin_max = max([data(:,i);previous_data(:,i)]) + eps;
            bin_gap = (bin_max - bin_min) / (bin_num-1);
            
            for j=1:bin_num
                if j~=bin_num
                    p(j) = sum(data(:,i) >= bin_min + (j-1) * bin_gap & data(:,i) < bin_min + j * bin_gap);
                    q(j) = sum(previous_data(:,i) >= bin_min + (j-1) * bin_gap & previous_data(:,i) < bin_min + j * bin_gap);
                else
                    p(j) = sum(data(:,i) >= bin_min + (j-1) * bin_gap & data(:,i) <= bin_max);
                    q(j) = sum(previous_data(:,i) >= bin_min + (j-1) * bin_gap & previous_data(:,i) <= bin_max);
                end
            end
            p = p / sum(p);
            q = q / sum(q);
            pq(i) = sqrt(sum((sqrt(p) - sqrt(q)).^2));
        end
        
        fea_set_num = length(fea_idx);
        for fea_i = 1:fea_set_num
            ws(fea_i,1) = 1 - (mean(pq(fea_idx{fea_i}))) / sqrt(2);
        end
        
        alpha = (ws + wd) / 2;
        for fea_i = 1:fea_set_num
            pred(:,fea_i) = predict_base(data(:,fea_idx{fea_i}),model{fea_i});
        end
        pred = pred * alpha;
        
    end
    
    function model = LearnH(data, label, T, ts)
    
        pos_idx = label==1;
        neg_idx = label==-1;
        pos_data = data(pos_idx,:);
        neg_data = data(neg_idx,:);
        pos_num = sum(pos_idx);
        
        w = importance_sampling(pos_data, ts);
        
        for t=1:T
            sampling_pos_data = datasample(pos_data,pos_num,'Weights',w);
            sampling_neg_data = datasample(neg_data,pos_num);
            sampling_data = [sampling_pos_data;sampling_neg_data];
            sampling_label = [ones(pos_num,1);-ones(pos_num,1)];
            model{t} = fitctree(sampling_data,sampling_label);
        end
        
    end

    function w = importance_sampling(data, ts)
        
        t = max(ts);
        l = min(ts);
        data_num = size(data,1);
        for k=l:t
            D(k) = sum(ts==k);
        end
        for k=l:t
            for j=1:size(data,2)
                u(k,j) = sum(data(ts==k,j)) / sum(ts==k);
                v(k,j) = sum((data(ts==k,j) - u(k,j)).^2) / (sum(ts==k)-1);
            end
        end
        
        for i=1:data_num
            gamma = 1;
            for j=1:size(data,2)
                k = ts(i);
                Dk = 1 / sqrt(2*pi*v(k,j)) * exp(-(data(i,j)-u(k,j))^2 / (2*v(k,j)));
                Dt = 1 / sqrt(2*pi*v(t,j)) * exp(-(data(i,j)-u(t,j))^2 / (2*v(t,j)));
                gamma = gamma * Dk / Dt;
            end
            beta = 1 / (D(ts(i)) / D(t) * gamma);
            w(i) = 1 / (1 + exp(-(beta-0.5)));
        end
    end
    

    function pred = predict_base(data,model)


        T = length(model);

        for t=1:T
            pred_single(:,t) = predict(model{t},data);
        end
        pred = sign(sum(pred_single,2));


    end