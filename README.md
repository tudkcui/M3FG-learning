# Major-Minor Mean Field Games
This repository is the official implementation of the paper "Learning Discrete-Time Major-Minor Mean Field Games".

### Install Python packages in virtual env
To install a virtual environment with the required packages, run the following.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run experiments
You can run experiments by calling main_fp.py. Options can be found in args_parser.
```
python main_fp.py
```

### Replicate results
To run all the experiments, run the experiment scripts.
```
python exp_1.py
python exp_2.py
python exp_3.py
```

### Replicate plots
After running the experiments, the plots in the paper can be generated using the plot scripts in the eval/ folder, 
or the following script to plot all.
```
python plot_all.py
```
