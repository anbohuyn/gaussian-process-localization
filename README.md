# Indoor Localization using Gaussian Processes

Pre-requisites:-
1. Ubuntu 16.04+ (Tested on 16.04 LTS)
2. Python 2.7

Install Dependencies:- 
1. numpy
2. matplotlib
3. scikit-learn

Setup instructions:-
1. Run InstallDependencies .sh (Elevated accces required, alternative InstallDependenciesUser.sh)

Runnning Code:-
1. python ./source/main.py - (Run main.py from terminal)


Debugging steps (In case data is missing/empty)

How to get Data files :-
1. Download http://robosrv.ucmerced.edu/public/software/IROS2012/UCM-WiFi-Localizer_Processed-Data.zip
2. Unzip "contents" to this location ./data/UCM_data/


Preprocessing Data :-
1. Run python ./source/read_UCM_data.py
2. This shoudl create processed numpy data in ./data/UCM_data/generated/ directory 


For any issues, reach out to anouni@cs.umass.edu
