This code is for the LdSM algorithm proposed in our research paper titled “LdSM: Logarithm-depth Streaming 
Multi-label Decision Trees” authored by Maryam Majzoubi and Anna Choromanska which was published at AISTATS 2020. 
The code is written in C++ and should compile on 64 bit Windows/Linux machines using a C++11 enabled compiler. 

Data sets
---------------------------------------
For downloading the benchmark multi-label data sets please visit the Extreme Classification Repository 
(http://manikvarma.org/downloads/XC/XMLRepository.html). The data format required for LdSM is different than the 
original data sets. In order to format the original data please use the following functions in MATLAB (note that
they require MATLAB tools from the above repo):

For Mediamill, Bibtex, Delicious data sets: format_data_split(data_filename, trfilename, tstfilename, save_name)
For the rest of data sets: format_data(train_filename, test_filename, save_name)
Where save_name is the name for saving the formatted data, and the rest of the files can be downloaded from the above repo.
Example for Delicious data set:
format_data_split('Delicious_data.txt', 'delicious_trSplit.txt', 'delicious_tstSplit.txt', 'delicious')

Compiling
---------------------------------------
mkdir build && cd build && cmake .. && make

Running experiments
---------------------------------------
We put a toy example data set which you can run using the following:
mkdir results && cd scripts && ./run.sh
