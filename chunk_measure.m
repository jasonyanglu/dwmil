function [ result ] = chunk_measure(  pred_value, label, chunk_num  )
%CHUNK_MEASURE Summary of this function goes here
%   Detailed explanation goes here
    
   
    for chunk_i=1:chunk_num
        
        [result.auc(chunk_i),result.gm(chunk_i)] = ...
            ImbalanceEvaluate(pred_value{chunk_i}, label{chunk_i});
        
    end

end



    function [auc,gm]=ImbalanceEvaluate(pred_value, label)


        
        pred = sign(pred_value);
        pred(pred==0) = 1;
        
        tp=sum(label==1 & pred==1);
        fn=sum(label==1 & pred==-1);
        tn=sum(label==-1 & pred==-1);
        fp=sum(label==-1 & pred==1);
        

        pos_rec = tp/(tp+fn);
        neg_rec = tn/(tn+fp);
        
        gm=sqrt(pos_rec*neg_rec);


        [~,~,~,auc] = perfcurve(label,pred_value,1);
    
    end