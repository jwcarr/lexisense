Orthographic informativity and the optimal viewing position
===========================================================

This repo contains the analytical code and supporting data for a project on orthographic informativity and the optimal viewing position. The top-level structure of the repo is:

- `code/`: Python code used for the analysis.

- `data/`: Various unprocessed and processed data files explained in more detail below.

- `experiment/`: Experimental code.

- `manuscript/`: LaTeX source and postscript figures for the manuscript.

- `visuals/`: Various visualizations and illustrations.


Analysis code
-------------

The Python files in `code/` generally follow these conventions:

- Files beginning with `bacs`: Code relating to BACS characters

- Files beginning with `exp`: Code for processing the experimental data

- Files beginning with `fig`: Code for making the figures

- Files beginning with `model`: Code relating to the Bayesian model reader

- Files beginning with `typ`: Code relating to the typological analyses


Data
----

- `data/corpora/`: Placeholder directory – original files not included in this repo

- `data/experiments`: Experimental parameter files and raw participant data

- `data/subtlex/`: Placeholder directory – original files not included in this repo

- `data/typ_uncertainty/`: Pickled uncertainty estimates for the sample languages

- `data/typ_word_probs/`: Pickled word probabilities for the sample languages


Experimental code
-----------------

### Online experiments

The code for the online experiments is located in `experiments/online/`. If you just have some technical questions about the design, you may be able to find answers in `server.js` or `client.js`, which contain most of the experimental code. If you actually want to run the experiment, you will first need to install [Node.js](https://nodejs.org) and [MongoDB](https://www.mongodb.com) on your system. Once installed, `cd` into the experiment directory and install the required node modules:

```shell
$ cd experiments/online/
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

replacing protocol, domain, and port with the appropriate strings (e.g., `http://localhost:8080`). Initially, you will see "No task available", since no task has yet been added to the database. The tasks are defined in JSON files in `data/experiments/online/`. To launch one, run e.g.:

```shell
python mission_control.py --launch exp1_left
```

After refreshing the browser, you should now be able to access the experiment.


License
-------

Except where otherwise noted, this repository is licensed under a Creative Commons Attribution 4.0 license. You are free to share and adapt the material for any purpose, even commercially, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. See LICENSE.md for full details.
