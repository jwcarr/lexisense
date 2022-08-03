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
# experiment.print_test_accuracy()
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
# 	plots.plot_learning_scores(fig[0,0], experiment)
# 	plots.plot_learning_curve(fig[0,1], experiment, n_previous_trials=1)
# 	plots.plot_test_curve(fig[1,0], experiment.left, show_individuals=True)
# 	plots.plot_test_curve(fig[1,1], experiment.right, show_individuals=True)
# 	plots.draw_brace(fig[1,0], (2,4), 0.2, 'High\ninformation\ncontent')
# 	plots.draw_brace(fig[1,0], (6,7), 0.2, 'Low\ninformation\ncontent')
# 	plots.draw_brace(fig[1,1], (4,6), 0.2, 'High\ninformation\ncontent')
# 	plots.draw_brace(fig[1,1], (1,2), 0.2, 'Low\ninformation\ncontent')
# 	fig.auto_deduplicate_axes = False
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

# trace = experiment.get_posterior(ovp.MODEL_FIT / 'exp1_posterior.nc')
# table = arviz.summary(trace, hdi_prob=0.95)
# print(table[['mean', 'hdi_2.5%', 'hdi_97.5%']].to_latex())

# trace = experiment.get_posterior(ovp.MODEL_FIT / 'exp1_posterior_left.nc')
# table = arviz.summary(trace, hdi_prob=0.95)
# print(table[['mean', 'hdi_2.5%', 'hdi_97.5%']].to_latex())

# trace = experiment.get_posterior(ovp.MODEL_FIT / 'exp1_posterior_right.nc')
# table = arviz.summary(trace, hdi_prob=0.95)
# print(table[['mean', 'hdi_2.5%', 'hdi_97.5%']].to_latex())

# trace = experiment.get_posterior(ovp.MODEL_FIT / 'exp1_posterior_uniform.nc')
# table = arviz.summary(trace, hdi_prob=0.95)
# print(table[['mean', 'hdi_2.5%', 'hdi_97.5%']].to_latex())
##############################################################################


'''
Plot the priors and posteriors for each condition independently and for the
experiment as a whole. transform_to_param_bounds is set to True because the
beta priors are expressed in [0,1] space and need to be transformed to the
appropriate parameter bounds.
'''
##############################################################################
# from ovp import plots

# file_path = ovp.RESULTS/'exp1'/'posteriors_spike.pdf'
# with ovp.Figure(file_path, n_cols=2, n_rows=2, width=150) as fig:
# 	for param, axis in zip(['α', 'β', 'γ', 'ε'], fig):
# 		plots.plot_prior(axis, experiment, param, transform_to_param_bounds=True)
# 		plots.plot_posterior(axis, experiment, param)
# 		# plots.plot_posterior(axis, experiment, param, posterior_file=ovp.MODEL_FIT / 'exp1_posterior_uniform.nc', linestyle=':')
# 		# plots.plot_posterior(axis, experiment.left, param, posterior_file=ovp.MODEL_FIT / 'exp1_posterior_left.nc')
# 		# plots.plot_posterior(axis, experiment.right, param, posterior_file=ovp.MODEL_FIT / 'exp1_posterior_right.nc')
# 	fig.auto_deduplicate_axes = False
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

# file_path = ovp.RESULTS/'exp1'/'predictive.pdf'
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

# file_path = ovp.RESULTS/'exp1'/'uncertainty.pdf'
# with ovp.Figure(file_path) as fig:
# 	plots.plot_uncertainty(fig, uncertainty_left, color=experiment.left.color)
# 	plots.plot_uncertainty(fig, uncertainty_right, color=experiment.right.color)
# 	fig[0,0].set_ylim(0, 1)
##############################################################################


'''
Plot uncertainty draws. Same as above, except use draws from the posterior
predictive instead of the mean parameter estimates.
'''
##############################################################################
# from ovp import model_fit, plots

# uncertainty_left, uncertainty_right = model_fit.uncertainty_curves_from_posterior_draws(experiment, n_draws=100, n_sims=10000)

# file_path = ovp.RESULTS/'exp1'/'uncertainty_draws.pdf'
# with ovp.Figure(file_path) as fig:
# 	for draw_i in range(100):
# 		plots.plot_uncertainty(fig, uncertainty_left[draw_i], color=experiment.left.light_color, show_guidelines=False, linewidth=0.5)
# 	for draw_i in range(100):
# 		plots.plot_uncertainty(fig, uncertainty_right[draw_i], color=experiment.right.light_color, show_guidelines=False, linewidth=0.5)
# 	fig[0,0].set_ylim(0, 1)
##############################################################################
