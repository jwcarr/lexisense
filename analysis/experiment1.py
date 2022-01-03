'''
This script is the main entry point for the analysis of Experiment 1. To
reproduce the analyses, simply uncomment the particular code blocks below
that you are interested in. Then, if you want to dig into the details, look
up the relevant functions in ovp/experiment.py

Import the ovp package, load the Experiment 1 data into an Experiment object
'''
##############################################################################
import ovp

experiment = ovp.Experiment('exp1')
##############################################################################


'''
Set the exclusion theshold to 7/8. Participants who scored less than 7 in the
final 8 mini tests will be excluded from analysis.
'''
##############################################################################
experiment.set_exclusion_threshold(7, 8)
##############################################################################


'''
Specify the parameter bounds and priors that are relevant to this experiment.
In this case, we will use beta distributions that express our certainty about
the model parameters before running the experiment. These beta distributions
in [0,1] will be transformed to the relevant parameter bounds under the
hood.
'''
##############################################################################
experiment.set_params({
	'α': ( 0.0625, 0.9999),
	'β': ( 0.0001, 1.0000),
	'γ': (-0.9999, 0.9999),
	'ε': ( 0.0001, 0.9999),
})

experiment.set_priors({
	'α': ('beta', (8, 2)),
	'β': ('beta', (2, 8)),
	'γ': ('beta', (4, 2)),
	'ε': ('beta', (2, 16)),
})
##############################################################################


'''
Print various pieces of information about the participants. These print outs
include participants who have been excluded.
'''
##############################################################################
# experiment.print_comments()
# experiment.print_median_completion_time(use_first_trial_time=True)
# experiment.print_median_bonus()
##############################################################################


'''
Make plots of every participant's results (learning curve and test curve).
'''
##############################################################################
# from ovp import plots

# for participant in experiment.iter_with_excludes():
# 	file_name = f'{participant.task_id}_{participant.ID}.pdf'
# 	file_path = ovp.RESULTS/'exp1'/'individual_results'/file_name
# 	with ovp.Figure(file_path, n_cols=2, width=150) as fig:
# 		plots.plot_learning_curve(fig[0,0], participant)
# 		plots.plot_test_curve(fig[0,1], participant)
##############################################################################


'''
Make plots of the overall experimental data.
'''
##############################################################################
# from ovp import plots

# with ovp.Figure(ovp.RESULTS/'exp1'/'learning_scores.pdf', width=150) as fig:
# 	plots.plot_learning_scores(fig, experiment)

# with ovp.Figure(ovp.RESULTS/'exp1'/'learning_curves.pdf', width=150) as fig:
# 	plots.plot_learning_curve(fig, experiment)

# with ovp.Figure(ovp.RESULTS/'exp1'/'test_curves.pdf', width=150) as fig:
# 	plots.plot_test_curve(fig, experiment)
##############################################################################


'''
Make the Experiment 1 results figure for the manuscript. This combines a bunch
of the above plots into a single figure.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.FIGS/'exp1_results.eps'
# with ovp.Figure(file_path, n_rows=2, n_cols=2, width='double', height=100) as fig:
# 	plots.plot_learning_scores(experiment, fig[0,0])
# 	plots.plot_learning_curve(experiment, fig[0,1], n_previous_trials=1)
# 	plots.plot_test_curve(experiment.left, fig[1,0])
# 	plots.plot_test_curve(experiment.right, fig[1,1])
# 	plots.draw_brace(fig[1,0], (2,4), 0.2, 'High\ninformation\ncontent')
# 	plots.draw_brace(fig[1,0], (6,7), 0.2, 'Low\ninformation\ncontent')
# 	plots.draw_brace(fig[1,1], (4,6), 0.2, 'High\ninformation\ncontent')
# 	plots.draw_brace(fig[1,1], (1,2), 0.2, 'Low\ninformation\ncontent')
# 	fig.auto_deduplicate_axes = False
##############################################################################


'''
Create the surrogate likelihoods for use in the model fit procedure. Once
created, the surrogates are written to the data/model_fit directory.

This takes a long time to run, so only run this if you want to reproduce the
surrogate likelihoods from scratch.
'''
##############################################################################
# from ovp import model_fit

# params = {
# 	'n_evaluations': 500,
# 	'n_random_evaluations': 199,
# 	'n_simulations': 100000,
# }

# model_fit.create_surrogate_likelihood(experiment.left, **params)
# model_fit.create_surrogate_likelihood(experiment.right, **params)
# model_fit.create_surrogate_likelihood(experiment, **params)
##############################################################################


'''
Fit the model parameters by combining the likelihoods precomputed above with
the prior specified at the top of this script. As a sanity check, we first
compute the posteriors for each condition to check they are broadly the same,
then we compute the posterior for the experimental data as a whole.

This will take a little while to run, so only run this if you want to
reproduce the posteriors from scratch. Alternatively, turn down the number of
chains/samples.
'''
##############################################################################
# from ovp import model_fit

# params = {
# 	'n_samples': 30000,
# 	'n_tuning_samples': 1000,
# 	'n_chains': 4,
# }

# model_fit.fit_posterior(experiment.left, **params)
# model_fit.fit_posterior(experiment.right, **params)
# model_fit.fit_posterior(experiment, **params)
##############################################################################


'''
Use ArviZ to print posterior parameter estimates and credible intervals for
each condition independently and the experimental data as a whole.

You could also use ArviZ's plotting functions here to explore the results in
other ways.

All ess_bulk > 10,000 and r_hat = 1, so the sampler convergence look fine. The
estimates for the left and right conditions are pretty similar, suggesting
that it's safe to adopt the estimates from the full dataset as canonical.
'''
##############################################################################
# import arviz

# trace = experiment.left.get_posterior()
# print(arviz.summary(trace, hdi_prob=0.95))

# trace = experiment.right.get_posterior()
# print(arviz.summary(trace, hdi_prob=0.95))

# trace = experiment.get_posterior()
# print(arviz.summary(trace, hdi_prob=0.95))
##############################################################################


'''
Plot the priors and posteriors for each condition independently and for the
experiment as a whole. unnormalize is set to True because the posterior
parameter values are stored in [0,1] and need to be transformed back to the
original parameter bounds.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.RESULTS/'exp1'/'posteriors.pdf'
# with ovp.Figure(file_path, n_cols=2, n_rows=2, width=150) as fig:
# 	for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
# 		plots.plot_prior(axis, experiment, param, unnormalize=True)
# 		plots.plot_posterior(axis, experiment.left, param, unnormalize=True)
# 		plots.plot_posterior(axis, experiment.right, param, unnormalize=True)
# 		plots.plot_posterior(axis, experiment, param, unnormalize=True)
##############################################################################


'''
Make the Experiment 1 posteriors figure for the manuscript. This is the same
as above but sized appropriately for the manuscript.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.FIGS/'exp1_posteriors.eps'
# with ovp.Figure(file_path, n_cols=4, width='double', height=40) as fig:
# 	for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
# 		plots.plot_prior(axis, experiment, param, unnormalize=True)
# 		plots.plot_posterior(axis, experiment.left, param, unnormalize=True)
# 		plots.plot_posterior(axis, experiment.right, param, unnormalize=True)
# 		plots.plot_posterior(axis, experiment, param, unnormalize=True)
##############################################################################


'''
Plot posterior predictive checks for Experiment 1. First, we simulate 100
datasets based on the posterior parameter estimated from the experimental
data as a whole. Specifically, we draw a set of parameter values from the
posterior and instantiate N Readers with that set of values (N = number of
participants after excludes, which is inferred automatically from the
Experiment object). We then subject each Reader to the test (each word is
tested in each fixation position) and record all the trials in a big dataset.
Each dataset contains trials of the form (0, 3, 4, 3), correspondinng to
(lexicon_index, target word, position, inferred object).

Then, we plot the mean test curve for each of these simulated datasets and
also the mean test curve for the actual experimental results. If the model
and model parameter estimates are okay, the actual experimental results
should fall within the posterior predictive distribution.
'''
##############################################################################
# from ovp import model_fit, plots

# sim_datasets = model_fit.simulate_from_posterior(experiment, n_sims=100)

# file_path = ovp.FIGS/'exp1_posterior_predictive.eps'
# with ovp.Figure(file_path, n_cols=2, width='double', height=50) as fig:
# 	plots.plot_posterior_predictive(fig[0,0], sim_datasets, experiment.left, lexicon_index=0, show_legend=True)
# 	plots.plot_posterior_predictive(fig[0,1], sim_datasets, experiment.right, lexicon_index=1)
##############################################################################


'''
Plot the uncertainty curve for each lexicon, taking into account the asymmetry
in the visual span. To do this, we instantiate a Reader with parameter values
estimated from the experiment and compute uncertainty over letter positions.
This plot reveals that uncertainty is not mirrored across the two lexicons;
uncertainty is much more evenly distributed in the right-heavy lexicon due to
the interaction from the visual span.
'''
##############################################################################
# from ovp import model_fit, plots

# uncertainty_left, uncertainty_right = model_fit.uncertainty_curve_from_posterior(experiment, n_sims=10000)

# file_path = ovp.FIGS/'exp1_predicted_uncertainty.eps'
# with ovp.Figure(file_path) as fig:
# 	plots.plot_uncertainty(fig, uncertainty_left, color=experiment.left.color)
# 	plots.plot_uncertainty(fig, uncertainty_right, color=experiment.right.color)
# 	fig[0,0].set_ylim(0, 1)
##############################################################################
