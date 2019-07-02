function wekaOBJ = matlab2weka_Instances(name, featureNames, data, label, targetIndex)
% Convert matlab data to a weka java Instances object for use by weka
% classes. 
%
% name           - A string, naming the data/relation
%
% featureNames   - A cell array of d strings, naming each feature/attribute
%
% data           - An n-by-d matrix with n, d-featured examples or a cell
%                  array of the same dimensions if string values are
%                  present. You cannot mix numeric and string values within
%                  the same column. 
%
% wekaOBJ        - Returns a java object of type weka.core.Instances
%
% targetIndex    - The column index in data of the target/output feature.
%                  If not specified, the last column is used by default.
%                  Use the matlab convention of indexing from 1.
%
% Written by Matthew Dunham

%     if(~wekaPathCheck),wekaOBJ = []; return,end
    if(nargin < 4)
        targetIndex = numel(featureNames); %will compensate for 0-based indexing later
    end
    
    % converting to nominal variables (Weka cannot classify numerical classes)
    train_label_nom = cell(size(label));
    ulabel = unique(label);
    tmp_cell = cell(1,1);
    for i = 1:length(ulabel)
        tmp_cell{1,1} = strcat('class_', num2str(i-1));
%         tmp_cell{1,1} = strcat('0', num2str(ulabel(i));
        label_nom(label == ulabel(i),:) = repmat(tmp_cell, sum(label == ulabel(i)), 1);
    end

    import weka.core.*;
    vec = FastVector();
%     if(iscell(data))
        for i=1:numel(featureNames)-1
%             if(ischar(data{1,i}))
                
%             else
                vec.addElement(Attribute(featureNames{i})); 
%             end
        end 
        attvals = unique(label_nom);
        values = FastVector();
        for j=1:numel(attvals)
           values.addElement(attvals{j});
        end
        vec.addElement(Attribute(featureNames{numel(featureNames)},values));
%     else
%         for i=1:numel(featureNames)
%             vec.addElement(Attribute(featureNames{i})); 
%         end
%     end
    wekaOBJ = Instances(name,vec,size(data,1));
%     if(iscell(data))
        for i=1:size(data,1)
            inst = DenseInstance(numel(featureNames));
            for j=0:numel(featureNames)-2
               inst.setDataset(wekaOBJ);
               inst.setValue(j,data(i,j+1));
            end
            inst.setValue(numel(featureNames)-1,label_nom{i});
            wekaOBJ.add(inst);
        end
%     else
%         for i=1:size(data,1)
%             inst = DenseInstance(1,data(i,:));
%             inst.setDataset(wekaOBJ);
%             inst.setValue(numel(featureNames)-1,i);
%             wekaOBJ.add(inst);
%         end
%     end
    wekaOBJ.setClassIndex(targetIndex-1);
end