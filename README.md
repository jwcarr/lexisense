Orthographic informativity and the optimal viewing position
===========================================================

This repo contains the analytical code and supporting data for a project on orthographic informativity and the optimal viewing position. The top-level structure of the repo is:

- `analysis/`: Main top-level analysis scripts.

- `data/`: Various unprocessed and processed data files explained in more detail below.

- `experiments/`: Experimental code.

- `manuscript/`: LaTeX source and postscript figures for the manuscript.

- `ovp/`: Python package containing the core analysis code.


Data
----

The `data` directory contains the following subdirectories: 

- `data/corpora/`: Placeholder directory – original files not included in this repo

- `data/experiments`: Experimental parameter files and raw participant data

- `data/subtlex/`: Placeholder directory – original files not included in this repo

- `data/typ_uncertainty/`: Pickled uncertainty estimates for the sample languages

- `data/typ_word_probs/`: Pickled word probabilities for the sample languages


Analysis code
-------------

All of our analysis code was written for Python 3.9. If you have an earlier version of Python, it may first be necessary to install a newer version. Once you have a working copy of Python 3.9 or newer, clone or download this repository and `cd` into the top-level directory:

```shell
$ cd path/to/ovp/
```

It is necessary to install various Python packages. The exact version numbers we used are documented in `requirements.txt`. To ensure that these packages/versions do not interfer with your own projects, you will probably want to create and activate a new Python virtual environment, for example:

```shell
$ python3 -m venv ovp_env
$ source ovp_env/bin/activate
```

Wtih the new environment activated, install the required Python packages from `requirements.txt`:

```shell
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Finally, the `ovp` subdirectory, which contains the main codebase, must be "installed" so that it can be accessed by the main analysis scripts in the `analysis` directory:

```shell
pip install -e .
```

It should now be possible to run the main scripts, for example:

```
$ python analysis/exp1.py
```

Read the scripts for more details, and, if you need to dig into the code in more depth, check the code under the `ovp` directory.


Experimental code
-----------------

### Experiment 1

The code for Experiment 1 (online experiment) is located in `experiments/exp1/`. If you just have some technical questions about the design, you may be able to find answers in `server.js` or `client.js`, which contain most of the experimental code. If you actually want to run the experiment, you will first need to install [Node.js](https://nodejs.org) and [MongoDB](https://www.mongodb.com) on your system/server. Once installed, `cd` into the experiment directory and install the required node modules:

```shell
$ cd experiments/exp1/
$ npm install
```

You will also need to make sure MongoDB is running, e.g.:

```shell
$ mkdir db
$ mongod -dbpath db
```

In `server.js`, set `PROTOCOL` to `http` (easier for testing) or `https` (secure), and set `PORT` to an open port number (you may need to open this port in a firewall). If you are using https, you will also need to provide the paths to the relevant encryption keys. If everything is set up correctly, you should be able to launch the server:

```shell
$ node server.js
```

In a browser, navigate to:

```
protocol://domain:port/?PROLIFIC_PID=000000000000000000000001
```

replacing protocol, domain, and port with the appropriate strings (e.g., `http://localhost:8080`). Initially, you will see "No task available", since no task has yet been added to the database. The tasks are defined in JSON files in `data/experiments/`. To launch one, run e.g.:

```shell
python mission_control.py --launch exp1_left
```

After refreshing the browser, you should now be able to access the experiment.


### Experiment 2

The code for Experiment 2 (eye tracking lab experiment) is located in `experiments/exp2/`. If you just have some technical questions about the experiment, you should find answers in `main.py`.

If you actually want to run the experiment, then first note that our experiment code was written for Python 3.6 and uses the PsychoPy package. Therefore, to run this code, it is a good idea to create a separate virtual environment to create a self-contained place for all PsychoPy's dependencies, for example:

```shell
$ cd experiments/exp2/
$ python3 -m venv ovp_exp2_env
$ source ovp_exp2_env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

The experiment can then be run with a command like:

```shell
python main.py exp2_left 99
````

where `exp2_left` is a task ID and `99` is a participant ID. Check the code for more details.


License
-------

Except where otherwise noted, this repository is licensed under a Creative Commons Attribution 4.0 license. You are free to share and adapt the material for any purpose, even commercially, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. See LICENSE.md for full details.
