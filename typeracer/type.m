%% ========== Typeracer Practice ==========
% This project is meant to demonstrate a high level understanding of the basic
% principles that Machine Learning utilizes. I am using my own typeracer.com
% dataset which contains a history of my typing speed in WPM on a variety of
% short passages. The data is as listed below:
% Race #
% WPM
% Accuracy
% Rank
% # Racers
% Text ID
% Date/Time (UTC)

%%
% Read in data
raceData = csvread('race_data.csv');
