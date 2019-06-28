function final_pred=cbce(x,y,a,b,c,d,e)
% this code is implemented by the authors of the paper "Online Ensemble Learning of Data Streams with Gradually Evolved Classes"

right=0;
wrong=0;

for i=1:size(x,1)
    x(i,:) = (x(i,:) - min(x(i,:))) / (max(x(i,:)) - min(x(i,:)));
end

%%online parameters
ratio_decay = e;
disappearance_threshold = 0.00001;

options.eta = a;
options.lamda = b;
KernelOptions.t = c;
options.cnt = 5000;

%%DDM
DDM_open = d; %true - 1; false - 0
JGAMAMETHOD_MINNUMINST = 30;

%%class info
class_exist = [];
class_ratio = [];
class_ratio_initial = [];

%%CB Model: alpha, trainFea, norm2X, currentAlpha, norm_ft
ensemble = [];

%%experiment result output
% fid_result = fopen(file_result, 'w');

%%initial data set
% x = [];     %feature values
% y = [];     %class labels
% load(file_data);
[dimension,example_count] = size(x);

%%read & classify example, update model
t_count=0;
tic;
while t_count<example_count
    t_count = t_count+1;
    if(mod(t_count,1000)==0)
        save(strcat('2DModel',num2str(1000)),'ensemble');
    end
    real_label = y(t_count);
    new_example = x(:,t_count);
    real_index = find(class_exist==real_label);
    
    %%classify examples
    classifier_count = length(ensemble);
    ft_array = [];
    if(classifier_count>0)
        classify_result = zeros(1, classifier_count);
        ft_array = zeros(1, classifier_count);
        if(DDM_open==1)
            tmp_p = zeros(1, classifier_count);
        end
        for i = 1 : classifier_count
            if(ensemble(i).active == 0)
                classify_result(i) = 0;
                ft_array(i) = NaN;
            else
                [classify_result(i),ft_array(i)] = OnlineKLRClassify(new_example,ensemble(i));
                if(DDM_open==1)
                    tmp_p(i) = ensemble(i).DDM_p;
                    if(i==real_index)
                        if(classify_result(i)>=0.5)
                            tmp_p(i) = tmp_p(i) - tmp_p(i)/ensemble(i).DDM_n;
                        else
                            tmp_p(i) = tmp_p(i) + (1-tmp_p(i))/ensemble(i).DDM_n;
                        end
                    else
                        if(classify_result(i)<=0.5)
                            tmp_p(i) = tmp_p(i) - tmp_p(i)/ensemble(i).DDM_n;
                        else
                            tmp_p(i) = tmp_p(i) + (1-tmp_p(i))/ensemble(i).DDM_n;
                        end
                    end
                end
            end
        end
        [max_probability, predic_subscript] = max(classify_result);
        prediction = class_exist(predic_subscript);
    else
        prediction = 0;
        max_probability = 0.5;
        if(DDM_open==1)
            tmp_p = [];
        end
    end
    final_pred(t_count) = prediction;
    if(real_label==prediction)
        right=right+1;
    else
        wrong=wrong+1;
    end
    %fprintf('%d %d %d %f\n', t_count,real_label, prediction, max_probability);
%     fprintf(fid_result, '%d %d %f\n', real_label, prediction, max_probability);

    %%update ratio and determine the class disappearance
    [class_exist, class_ratio, class_ratio_initial,class_disap,class_rec] = classRatioUpdate(class_exist, class_ratio, class_ratio_initial, real_label);
%     ratioTmp = zeros(1,4);
%     for i=1:length(class_ratio)
%         ratioTmp(i) = class_ratio(i);
%     end
%     fprintf(fid_result, '%f %f %f %f\n', ratioTmp(1), ratioTmp(2), ratioTmp(3),ratioTmp(4));
    if(~isempty(class_disap))
        for i=1:length(class_disap)
            ensemble(class_disap(i)).active = 0;
        end
    end
    if(class_rec~=0)
        ensemble(class_rec).active = 1;
    end

    %%update CB models
    real_subscript = find(class_exist==real_label);
    if(real_subscript == length(ensemble)+1)
       ensemble(real_subscript).currentAlpha = zeros(1,options.cnt);
       ensemble(real_subscript).norm2X = zeros(1,options.cnt);
       ensemble(real_subscript).trainFea = zeros(dimension,options.cnt);
       ensemble(real_subscript).index = 1;          %need update support vector
       ensemble(real_subscript).firstloop = 1;      %
       ensemble(real_subscript).active = 1;
       
       if(DDM_open == 1)
           ensemble(real_subscript).DDM_n = 1;
           ensemble(real_subscript).DDM_p = 1;
           ensemble(real_subscript).DDM_s = 0;
           ensemble(real_subscript).DDM_psmin = +inf;
           ensemble(real_subscript).DDM_pmin = +inf;
           ensemble(real_subscript).DDM_smin = +inf;
           ensemble(real_subscript).warnExample = zeros(dimension,1);
           ensemble(real_subscript).warnLabel = zeros(1,1);
           tmp_p(end+1) = 1;
       end
       
       ft_array(end+1) = 0;
    end
    classifier_count = length(ensemble);
    %fprintf(fid_result, '\n');
    for i = 1 : classifier_count
        label_tmp = 0;
        %tmp_p = ensemble(i).DDM_p;
        if(i == real_subscript)
            label_tmp = 1;                          %positive label            
        else
            ratio_tmp = class_ratio(real_subscript);
            random_num = rand();
            select_ratio = ratio_tmp/(1-ratio_tmp);
            if (random_num < select_ratio)
               label_tmp = -1;                      %negative label
            end
        end
        %fprintf(fid_result, '%f\t', label_tmp);
        
        if(DDM_open == 1)
            if(label_tmp~=0)
                ensemble(i).DDM_p = tmp_p(i);
                ensemble(i).DDM_s = sqrt(tmp_p(i)*(1-tmp_p(i))/ensemble(i).DDM_n);
                ensemble(i).DDM_n = ensemble(i).DDM_n + 1;
                if(ensemble(i).DDM_p + ensemble(i).DDM_s <= ensemble(i).DDM_psmin)
                    ensemble(i).DDM_pmin = ensemble(i).DDM_p;
                    ensemble(i).DDM_smin = ensemble(i).DDM_s;
                    ensemble(i).DDM_psmin = ensemble(i).DDM_p + ensemble(i).DDM_s;                   
                elseif((ensemble(i).DDM_n >= JGAMAMETHOD_MINNUMINST) && (ensemble(i).DDM_p + ensemble(i).DDM_s > ensemble(i).DDM_pmin + 9*ensemble(i).DDM_smin))
                    fprintf('%d : drift\n',i);
                    fprintf('%f %f',ensemble(i).DDM_p,ensemble(i).DDM_s);
                    ensemble(i).currentAlpha = zeros(1,options.cnt);
                    ensemble(i).norm2X = zeros(1,options.cnt);
                    ensemble(i).trainFea = zeros(dimension,options.cnt);
                    ensemble(i).index = 1;          %need update support vector
                    ensemble(i).firstloop = 1;      %
                    ensemble(i).active = 1;
                    for k = 2:length(ensemble(i).warnLabel)
                        [~,ft_tmp] = OnlineKLRClassify(ensemble(i).warnExample(:,k),ensemble(i));
                        [param,new_alpha,new_norm] = OnlineKLRUpdate(ensemble(i).warnExample(:,k),ensemble(i).warnLabel(k),ft_tmp,ensemble(i));
                        ensemble(i).currentAlpha = param*ensemble(i).currentAlpha;
                        ensemble(i).currentAlpha(ensemble(i).index) = new_alpha;
                        ensemble(i).norm2X(ensemble(i).index) = new_norm;
                        ensemble(i).trainFea(:,ensemble(i).index) = ensemble(i).warnExample(:,k);
                        ensemble(i).index = ensemble(i).index+1;
                        if(ensemble(i).index>options.cnt)
                            ensemble(i).index = 1;
                            ensemble(i).firstloop = 0;
                        end
                    end
                    ensemble(i).DDM_n = 1;
                    ensemble(i).DDM_p = 1;
                    ensemble(i).DDM_s = 0;
                    ensemble(i).DDM_psmin = +inf;
                    ensemble(i).DDM_pmin = +inf;
                    ensemble(i).DDM_smin = +inf;
                    ensemble(i).warnExample = zeros(dimension,1);
                    ensemble(i).warnLabel = zeros(1,1);
                elseif(ensemble(i).DDM_p + ensemble(i).DDM_s > ensemble(i).DDM_pmin + 7*ensemble(i).DDM_smin)
                    fprintf('%d : warn\n',i);
                    ensemble(i).warnExample(:,end+1) = new_example;
                    ensemble(i).warnLabel(end+1) = label_tmp;
                else
                    ensemble(i).warnExample = zeros(dimension,1);
                    ensemble(i).warnLabel = zeros(1,1);
                end
            end
        end
        
        if(label_tmp~=0)
            if(isnan(ft_array(i)))
                [~,ft_array(i)] = OnlineKLRClassify(new_example,ensemble(i));
            end
            [param,new_alpha,new_norm] = OnlineKLRUpdate(new_example,label_tmp,ft_array(i),ensemble(i));
            ensemble(i).currentAlpha = param*ensemble(i).currentAlpha;
            ensemble(i).currentAlpha(ensemble(i).index) = new_alpha;
            ensemble(i).norm2X(ensemble(i).index) = new_norm;
            ensemble(i).trainFea(:,ensemble(i).index) = new_example;
            ensemble(i).index = ensemble(i).index+1;
            if(ensemble(i).index>options.cnt)
                ensemble(i).index = 1;
                ensemble(i).firstloop = 0;
            end
        end
    end
end
time = toc;
% save(outname,'time');

    %%classify
    function [prob_yt_ft,ft_xt] = OnlineKLRClassify(xt,model)
        if(model.firstloop==1)
            T = model.index - 1;
        else
            T = options.cnt;
        end
        sigma=KernelOptions.t;

        norm2xt=sum(xt.*xt);

        % Depends on the kernel
        k_xt=construct_RBF_Row(norm2xt,model.norm2X(1:T),xt'*model.trainFea(:,1:T),sigma);
        ft_xt=k_xt*model.currentAlpha(1:T)';

        prob_yt_ft=LogisticProb(-ft_xt);
    end

    %update
    function [param,new_alpha,new_norm] = OnlineKLRUpdate(xt,yt,ft_xt,model)  
        % Columns are samples
        % Optimized for Gaussian Kernel
        if(model.firstloop==1)
            T = model.index - 1;
        else
            T = options.cnt;
        end
        sigma=KernelOptions.t;

        norm2xt=sum(xt.*xt);

        prob_yt_ft=LogisticProb(ft_xt*yt);
        tmp=options.eta*yt*(1-prob_yt_ft);
        %1:model.currentAlpha = (1-options.eta*options.lamda)*model.currentAlpha;
        param = (1-options.eta*options.lamda);
        new_alpha = tmp;
        %1:model.currentAlpha(end+1)=tmp;
        %2:model.norm2X(end+1)=norm2xt;
        new_norm = norm2xt;
    end

    %logist
    function prob = LogisticProb(value)
            prob = 1/(1+exp(value));
    end

    %RBF vector
    function k_xt = construct_RBF_Row(norm2xt, norm2X, xtTrainFea, sigma)
        xtx = norm2X + norm2xt - 2*xtTrainFea;
        k_xt = exp(xtx/(-2*sigma^2));
    end

    %class percentage ratio update
    function [class_exist, class_ratio, class_ratio_initial,class_disap,class_rec] = classRatioUpdate(o_class_exist, o_class_ratio, o_class_ratio_initial, current_class_label)
        current_class_subscript = find(o_class_exist==current_class_label, 1);
        class_count = length(o_class_exist);
        class_disap = [];
        class_rec = 0;

        class_exist = o_class_exist;
        class_ratio = o_class_ratio;
        class_ratio_initial = o_class_ratio_initial;

        %%update for the class that current example belongs to
        if (isempty(current_class_subscript))
            %novel class emergence
            current_class_subscript = class_count + 1;
            class_exist(end+1) = current_class_label;
            class_ratio(end+1) = 0;
            class_ratio_initial(end+1) = 1;
        elseif(class_ratio(current_class_subscript)==0)
            %recurrent class + second example needed for calculate ratio (recurrent or novel)
            if (class_ratio_initial(current_class_subscript)==0)
                %first reveive (recurrent)
                class_rec = current_class_subscript;
                class_ratio_initial(current_class_subscript) = 1;
            else
                %second receive (recurrent or novel)
                new_ratio = 1/class_ratio_initial(current_class_subscript);
                class_ratio(current_class_subscript) = new_ratio;
                class_ratio = class_ratio/sum(class_ratio);
                class_ratio(current_class_subscript) = ratio_decay*class_ratio(current_class_subscript)+1-ratio_decay;
                class_ratio_initial(current_class_subscript) = 0;
            end
        else
            %current exisiting class
            class_ratio(current_class_subscript) = ratio_decay*class_ratio(current_class_subscript)+1-ratio_decay;
        end

        %%update for the other classes
        for j = 1 : class_count
            if (current_class_subscript ~= j)
                %update ratio initial count
                if (class_ratio_initial(j) ~= 0)
                    class_ratio_initial(j) = class_ratio_initial(j)+1;
                end
                %update ratio percentage
                if (class_ratio(j) ~= 0)
                    class_ratio(j) = ratio_decay*class_ratio(j);
                    %set class disappearence
                    if (class_ratio(j) < disappearance_threshold)
                        class_ratio(j) = 0;
                        class_ratio_initial(j) = 0;
                        class_disap(end+1) = j;
                    end
                end
            end
        end
    end
end