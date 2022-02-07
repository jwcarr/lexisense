'''
This script is the main entry point for the analysis of Experiment 2. First,
some notes on the preprocessing of the eyetracking data. The experiment
script produces two files: 01.json and 01.edf. The JSON file stores all the
data about trial order, button clicks etc, and the EDF file stores the
eyetracker recording. Since EDF is a proprietary format, we first convert the
file to ASC using EyeLink's edf2asc tool, e.g.:

	$ edf2asc data/01.edf

From this ASC file, we will then extract the fixation sequence and store it
inside the JSON file to keep everything neatly organized together. The raw
ASC and EDF files are not committed to the repo because they are very large
(contact me if you want this raw data). The extraction process is run like
this:
'''
##############################################################################
# from ovp import merge_fixation_data

# merge_fixation_data.merge_fixations_into_user_data(json_path, asc_path)
##############################################################################
'''
This uses Eyekit to extract the fixations from the ASC file and then inserts
them into the JSON file. It also performs a number of checks to make sure the
recordings in the ASC file match the trials in the JSON file, and also
creates Eyekit TextBlock objects for each trial (also stored in the JSON).

Alternatively, the extraction process can be run from the command line, e.g.:

	$ python ovp/merge_fixation_data.py exp2_left 01

Note, however, that the JSON files committed to the repo have already been
processed, so it is not necessary to run any of these preprocessing steps
unless you need to reproduce the data for some reason.
'''


'''
Import the ovp package and load the Pilot 3 data into an Experiment object:
'''
##############################################################################
import ovp
experiment = ovp.Experiment('exp2')
##############################################################################


'''
Specify the parameter bounds and priors that are relevant to this experiment.
For this experiment, the parameter bounds do not enter into the statistical
model and are only used for setting the limits on the plots.
'''
##############################################################################
experiment.set_params({
	'τ': (0, 252),
	'δ': (0, 60),
	'ζ': (0, 60),
	'ξ': (0, 60),
})

experiment.set_priors({
	'ζ': ('exponential', (0.1,)),
	'ξ': ('exponential', (0.1,)),	
})

experiment.left.set_priors({
	'τ': ('normal', (72., 20.)),
	'δ': ('gamma', (20., 8.)),
})

experiment.right.set_priors({
	'τ': ('normal', (144., 20.)),
	'δ': ('gamma', (30., 8.)),
})
##############################################################################


'''
Set the exclusion threshold to 7/8. Participants who scored less than 7 in the
final 8 mini tests will be excluded from analysis. In addition, we will
manually exclude certain participants for the reasons noted below.
'''
##############################################################################
experiment.set_exclusion_threshold(7, 8)
##############################################################################


'''
Print various pieces of information about the participants. These print outs
include participants who have been excluded.
'''
##############################################################################
# experiment.print_median_completion_time(use_first_trial_time=False)
##############################################################################


'''
Make plots of every participant's results (learning curve and landing
positions).
'''
##############################################################################
# from ovp import plots

# for participant in experiment.iter_with_excludes():
# 	file_name = f'{participant.task_id}_{participant.ID}.pdf'
# 	file_path = ovp.RESULTS/'exp2'/'individual_results'/file_name
# 	with ovp.Figure(file_path, n_cols=2, width=150) as fig:
# 		plots.plot_learning_curve(fig[0,0], participant)
# 		plots.plot_landing_curve(fig[0,1], participant)
##############################################################################


'''
Create images of every participant's individual trials. The make_trial_image
function uses Eyekit to create the image and returns an Image object.
'''
##############################################################################
# from ovp import plots

# for participant in experiment.iter_with_excludes():
# 	file_path = ovp.RESULTS/'exp2'/'individual_results'/f'{participant.task_id}_{participant.ID}'
# 	if not file_path.exists():
# 		file_path.mkdir(parents=True)
# 	for i, trial in enumerate(participant.iter_free_fixation_trials()):
# 		img = plots.make_trial_image(participant, trial)
# 		img.save(file_path / f'{i}.pdf')
##############################################################################


'''
Make plots of the overall experimental results.
'''
##############################################################################
# from ovp import plots
	
# with ovp.Figure(ovp.RESULTS/'exp2'/'learning_scores.pdf', width=150) as fig:
# 	plots.plot_learning_scores(fig, experiment)

# with ovp.Figure(ovp.RESULTS/'exp2'/'learning_curves.pdf', width=150) as fig:
# 	plots.plot_learning_curve(fig, experiment)

# with ovp.Figure(ovp.RESULTS/'exp2'/'landing_curves.pdf', width=150) as fig:
# 	plots.plot_landing_curve(fig, experiment, show_average=True)
##############################################################################


'''
Make a plot of the landing curves to put include in the manuscript.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.FIGS/'exp2_results.eps'
# with ovp.Figure(file_path) as fig:
# 	plots.plot_landing_curve(fig, experiment, show_average=True)
# 	fig[0,0].set_ylabel(None)
##############################################################################


'''
Fit the statistical model from the landing position data. This uses the priors
set at the top of this script. This will take a little while to run, so only
run this if you want to reproduce the posteriors from scratch. Alternatively,
turn down the number of chains/samples.
'''
##############################################################################
# from ovp import landing_model

# params = {
# 	'n_samples': 5000,
# 	'n_tuning_samples': 1000,
# 	'n_chains': 8,
# }

# landing_model.fit_posterior(experiment, **params)
# landing_model.fit_posterior_with_separate_hyperpriors(experiment.left, **params)
# landing_model.fit_posterior_with_separate_hyperpriors(experiment.right, **params)
##############################################################################


'''
Use ArviZ to print posterior parameter estimates and credible intervals for
each condition. Not all ess_bulk > 10,000, but this is probably just due to
the small size of the dataset.

You could also use ArviZ's plotting functions here to explore the results in
other ways.
'''
##############################################################################
# import arviz

# trace = experiment.get_posterior()
# print(arviz.summary(trace, hdi_prob=0.95))
##############################################################################


'''
Plot the priors and posteriors. The results look very sensible: with just five
participants per condition, we already seem to have a nice effect.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.RESULTS/'exp2'/'posteriors.pdf'
# with ovp.Figure(file_path, n_cols=2, n_rows=2, width='double', height=100) as fig:
# 	plots.plot_prior(fig[0,0], experiment, 'τ')
# 	plots.plot_prior(fig[0,1], experiment, 'δ')
# 	plots.plot_posterior(fig[0,0], experiment, 'τ')
# 	plots.plot_posterior(fig[0,1], experiment, 'δ')
# 	plots.plot_posterior_difference(fig[1,0], experiment, 'τ', hdi=0.95, rope=(-9, 9))
# 	plots.plot_posterior_difference(fig[1,1], experiment, 'δ', hdi=0.95, rope=(-4, 4))
# 	plots.draw_letter_grid(fig[0,0], letter_width=36, n_letters=7)
##############################################################################
