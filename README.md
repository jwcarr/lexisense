Efficient eye movements in visual word recognition: Sensitivity to the structure of the lexicon
===============================================================================================

This repo contains the analytical code and supporting data for our ongoing project on how the structure of the lexicon influences eye movements. The top-level structure of the repo is:

- `data/`: Various unprocessed and processed data files

- `experiments/`: Code for Experiments 1 and 2

- `manuscript/`: LaTeX source and postscript figures for the manuscript

- `ovp/`: Python package containing the core analysis code

- `results/`: Various plots and visualizations


Data
----

The `data` directory contains the following subdirectories: 

- `data/corpora/`: Placeholder directory – original files not included in this repo

- `data/experiments`: Experimental parameter files and raw participant data

- `data/lang_uncertainty/`: JSON files containing uncertainty estimates for the sample languages

- `data/lang_word_probs/`: JSON files containing word probabilities for the sample languages

- `data/model_fit/`: NetCDF files containing precomputed posteriors

- `data/subtlex/`: Placeholder directory – original files not included in this repo


Analysis code
-------------

All of our analysis code was written for Python 3.9. If you have an earlier version of Python, it may first be necessary to install a newer version. Once you have a working copy of Python 3.9 or newer, clone or download this repository and `cd` into the top-level directory:

```bash
$ cd path/to/ovp/
```

The exact version numbers of the packages we used are documented in `requirements.txt`. To replicate our Python environment and ensure that the required packages do not interfere with your own projects, you will probably want to create and activate a new Python virtual environment, for example:

```bash
$ python3 -m venv ovp_env
$ source ovp_env/bin/activate
```

Wtih the new environment activated, install the required Python packages from `requirements.txt`:

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

If everything is installed and working correctly, you should be able to rebuild all the figures in the manuscript:

```bash
$ python make_figs.py
```

A Jupyter notebook (`notebook.ipynb`) is provided which guides you through recreating the basic results reported in the paper. If you want to dive into the data more thoroughly, I would recommend that you check the notebook first and then explore the code in the `ovp/` directory. To read the notebook, you will also need to install Jupyter Notebook:

```bash
$ pip install notebook
$ jupyter notebook
```


Experimental code
-----------------

### Experiment 1

The code for Experiment 1 (online experiment) is located in `experiments/exp1/`. If you just have some technical questions about the design, you may be able to find answers in `server.js` or `client.js`, which contain most of the experimental code. If you actually want to run the experiment, you will first need to install [Node.js](https://nodejs.org) and [MongoDB](https://www.mongodb.com) on your system/server. Once installed, `cd` into the experiment directory and install the required node modules:

```bash
$ cd experiments/exp1/
$ npm install
```

You will also need to make sure MongoDB is running, e.g.:

```bash
$ mkdir db
$ mongod -dbpath db
```

In `server.js`, set `PROTOCOL` to `http` (easier for testing) or `https` (secure), and set `PORT` to an open port number (you may need to open this port in a firewall). If you are using https, you will also need to provide the paths to the relevant encryption keys. If everything is set up correctly, you should be able to launch the server:

```bash
$ node server.js
```

In a browser, navigate to:

```
protocol://domain:port/?PROLIFIC_PID=000000000000000000000001
```

replacing protocol, domain, and port with the appropriate strings (e.g., `http://localhost:8080`). Initially, you will see "No task available", since no task has yet been added to the database. The tasks are defined in JSON files in `data/experiments/`. To launch one, run e.g.:

```bash
python mission_control.py --launch exp1_left
```

After refreshing the browser, you should now be able to access the experiment.


### Experiment 2

The code for Experiment 2 (eye tracking lab experiment) is located in `experiments/exp2/`. If you just have some technical questions about the experiment, you should find answers in `main.py`.

If you actually want to run the experiment, then first note that our experiment code was written for Python 3.6 and uses the PsychoPy package. Therefore, to run this code, it is a good idea to create a separate virtual environment to create a self-contained place for all PsychoPy's dependencies, for example:

```bash
$ cd experiments/exp2/
$ python3 -m venv ovp_exp2_env
$ source ovp_exp2_env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

The experiment can then be run with a command like:

```bash
python main.py exp2_left 99
````

where `exp2_left` is a task ID and `99` is a participant ID.


License
-------

Except where otherwise noted, this repository is licensed under a Creative Commons Attribution 4.0 license. You are free to share and adapt the material for any purpose, even commercially, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. See LICENSE.md for full details.
