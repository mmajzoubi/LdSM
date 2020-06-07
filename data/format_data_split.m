function format_data_split(filename, trFilename, tstFilename, save_name)

format long
[data,label]=read_data(filename);
trIndex = importdata(trFilename);
teIndex = importdata(tstFilename);
num = 3; % any number from 1-10
trIndex = trIndex(:,num); 
teIndex = teIndex(:,num);

trSize = size(trIndex,1);
tmp = randperm(trSize);
trIndex = trIndex(tmp);

testData = data(:,teIndex);
testLabel = label(:,teIndex);
trainData = data(:,trIndex);
trainLabel = label(:,trIndex);

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
 
