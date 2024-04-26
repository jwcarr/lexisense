Readers target words where they expect to minimize uncertainty
==============================================================

This repository contains the analytical code and supporting data for our project on how the structure of the lexicon influences eye movements. The paper describing this work is published in the *Journal of Memory and Language* and is [available here](https://doi.org/10.1016/j.jml.2024.104530).


tl;dr
-----

- For a walkthrough of all the core analyses, check `notebook.ipynb`.

- If you just want to get your hands on the experimental data, you should find what you need in `data/experiments/exp1.csv`, `data/experiments/exp2.csv`, and `data/experiments/exp3.csv`.

- If you want to play around with the cognitive model, you'll find this in `code/model.py`.

- To take a look at the statistical model, check `code/landing_model.py`.

- If you want to explore the experimental code, check `experiments/exp1/` (Node.js web app) and `experiments/exp2/` (PsychoPy code).


Organization
------------

The top-level structure of the repo is:

- `code/`: Python analysis code

- `data/`: Various unprocessed and processed data files

- `experiments/`: Code for Experiments 1, 2, and 3

- `manuscript/`: LaTeX source and postscript figures for the manuscript

The `data` directory contains the following subdirectories:

- `data/corpora/`: Placeholder directory – original files not included in this repo

- `data/experiments`: Experimental parameter files and raw participant data

- `data/lang_uncertainty/`: JSON files containing uncertainty estimates for the sample languages

- `data/lang_word_probs/`: JSON files containing word probabilities for the sample languages

- `data/model_fit/`: NetCDF files containing precomputed posteriors

- `data/subtlex/`: Placeholder directory – original files not included in this repo


Replicating the analysis
------------------------

A Jupyter notebook (`notebook.ipynb`) is provided which guides you through the basic results reported in the paper. I would recommend that you start by previewing this document here on GitHub. Then, to dive into full replication, I would recommend that you replicate my Python environment and work through the notebook, referring to the files under the `code/` directory as required. All analysis code was written for Python 3.10 – it may be necessary to upgrade if you have an earlier version.

To get started, clone or download this repository and `cd` into the top-level directory:

```bash
$ cd path/to/lexisense/
```

The exact version numbers of the Python packages we used are documented in `requirements.txt`. To replicate our Python environment and ensure that the required packages do not interfere with your own projects, you will probably want to create and activate a new Python virtual environment, for example:

```bash
$ python3 -m venv lexisense_env
$ source lexisense_env/bin/activate
```

With the new environment activated, install the required Python packages from `requirements.txt`:

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

If everything is installed and working correctly, the following command should give you a 👍:

```bash
$ python make_figs.py test
```

If you want to be able to interact with the notebook, you will also need to install Jupyter Notebook:

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
python mission_control.py exp1_left --launch https://app.prolific.co
```

providing the URL that participants should be redirected back to on completion of the experiment. After refreshing the browser, you should now be able to access the experiment.


### Experiment 2

The code for Experiment 2 (eye tracking lab experiment) is located in `experiments/exp2/`. If you just have some technical questions about the experiment, you should find answers in `main.py`.

If you actually want to run the experiment, then first note that our experiment code was written for Python 3.6 and uses the PsychoPy package. Therefore, to run this code, it is a good idea to create a separate virtual environment to create a self-contained place for all PsychoPy's dependencies, for example:

```bash
$ cd experiments/exp2/
$ python3 -m venv lexisense_env2
$ source lexisense_env2/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

The experiment can then be run with a command like:

```bash
python main.py exp2_left 99
```

where `exp2_left` is a task ID and `99` is a participant ID. The code is intended for use with a EyeLink 1000 eye tracker and requires SR Research's Pylink Python module (n.b. this is *not* the same Pylink package on PyPI/pip) and [EyeLinkCoreGraphicsPsychoPy](https://github.com/wanjam/Easy-Eyelink-Interface); however, by default the code is in "test mode", where the mouse cursor simulates the gaze position, and can therefore be tested without any of this infrastructure in place.


### Experiment 3

The Experiment 3 codebase is similar to that of Experiment 2, although we made some generalizations to the code to accommodate other possible experiments (that we never actually ran in the end). Nevertheless, the code we actually used is preserved here.


Citing this work
----------------

Carr, J. W., Fantini, M., Perrotti, L., & Crepaldi, D. (2024). Readers target words where they expect to minimize uncertainty. *Journal of Memory and Language*, *138*, Article 104530. https://doi.org/10.1016/j.jml.2024.104530

```bibtex
@article{Carr:2024,
author = {Carr, Jon W and Fantini, Monica, Perrotti, Lorena, and Crepaldi, Davide},
title = {Readers Target Words where They Expect to Minimize Uncertainty},
journal = {Journal of Memory and Language},
year = {2024},
volume = {138},
pages = {Article 104530},
doi = {10.1016/j.jml.2024.104530}
}
```
