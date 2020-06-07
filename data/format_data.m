function format_data(trFilename, tstFilename, save_name)

format long
[trainData,trainLabel]=read_data(trFilename);

[testData,testLabel]=read_data(tstFilename);
 
train_size = size(trainLabel,2);
tmp = randperm(train_size);
trainData = trainData(:,tmp);
trainLabel = trainLabel(:,tmp);

[atr,btr,str] = find(trainData);
[qtr,wtr] = find(trainLabel);

[ate,bte,ste] = find(testData);
[qte,wte] = find(testLabel);

deleteTheseIndices = ~ismember(btr, wtr);
btr(deleteTheseIndices) = [];
atr(deleteTheseIndices) = [];
str(deleteTheseIndices) = [];

deleteTheseIndices = ~ismember(bte, wte);
bte(deleteTheseIndices) = [];
ate(deleteTheseIndices) = [];
ste(deleteTheseIndices) = [];

dlmwrite([save_name '_trDataDim.csv'],atr,'precision','%.0f');
dlmwrite([save_name '_trDataPoint.csv'],btr,'precision','%.0f');
dlmwrite([save_name '_trDataValue.csv'],str);
 
dlmwrite([save_name '_teDataDim.csv'],ate,'precision','%.0f');
dlmwrite([save_name '_teDataPoint.csv'],bte,'precision','%.0f');
dlmwrite([save_name '_teDataValue.csv'],ste);
 
dlmwrite([save_name '_trLabelDim.csv'],qtr,'precision','%.0f');
dlmwrite([save_name '_trLabelPoint.csv'],wtr,'precision','%.0f');
 
dlmwrite([save_name '_teLabelDim.csv'],qte,'precision','%.0f');
dlmwrite([save_name '_teLabelPoint.csv'],wte,'precision','%.0f');
 
