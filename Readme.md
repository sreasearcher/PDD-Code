# Install
We recommend an installation with Git and Anaconda. 
### Download code

	git clone https://github.com/sreasearcher/ODD-Code.git
	cd ODD-Code
### Install necessary packages
	conda create -n ODD python=3.6
	conda activate ODD
	pip install numpy
	pip install matplotlib
### [Optional] Full installation

	pip install -r requirements.txt

# Run
For Fig. 8-10, run the following command three times with input parameters 54, 450, and 866.7, respectively.

	python exp_1_min.py

For Fig. 11-13, run the following command three times with input parameters 54, 450, and 866.7, respectively.

	python exp_2_window.py
