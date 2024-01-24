clc; clear; close all;

% load csv file
filename = 'EF_HH_V.csv';

% read file
freq_file = readtable(filename);

% convert to array
data = table2array(freq_file);
data = data(2:end,:);

% plot
hf = figure;
hf.Color = 'w';
hold on;

hist = histogram(data(:,2),36,'Normalization','pdf');
w1 = hist.BinEdges(2:end);
X1 = hist.Values;
xlim([0.1,3.6])

data = [w1',X1'];

filename = 'weibull_pdf_cook_inlet.xlsx';
writematrix(data, filename)



return