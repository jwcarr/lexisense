'''

This script is the main entry point for the analysis of Experiment 1. To
reproduce the analyses, simply uncomment the particular lines below that you
are interested in. Then, if you want to dig into the details, look up the
relevant functions in exp_analysis.py

'''

import core
import exp_analysis


##############################################################################
# Load the experiment data and set the exclusion threshold
##############################################################################
experiment = exp_analysis.Experiment('exp1')
experiment.set_exclusion_threshold(7, 8)


##############################################################################
# Print participant comments and other basic info
##############################################################################
# exp_analysis.print_comments(experiment)
# exp_analysis.calculate_median_completion_time(experiment)
# exp_analysis.calculate_median_bonus(experiment)
# exp_analysis.print_linguistic_backgrounds(experiment)
# exp_analysis.check_size_selections(experiment)


##############################################################################
# Make various plots
##############################################################################
# exp_analysis.plot_individual_results(core.VISUALS, tasks)
# exp_analysis.plot_learning_scores(core.VISUALS, tasks)
# exp_analysis.plot_learning_curves(core.VISUALS, tasks, show_individual_curves=True)
# exp_analysis.plot_ovp_curves(core.VISUALS/'test_plot.eps', tasks)
# exp_analysis.plot_test_inferences(core.VISUALS, tasks)


##############################################################################
# Make the results figure for the manuscript
##############################################################################
# exp_analysis.make_results_figure(experiment, core.FIGS/'exp1_results.eps')


##############################################################################
# Fit the model parameters to each condition independently and the entire
# dataset. This is slow - for testing purposes, turn down the number of
# simulations.
##############################################################################
# import model_fit

# model_fit.create_surrogate_likelihood(experiment.left, core.MODEL_FIT/'exp1_left_likelihood.pkl')
# model_fit.create_surrogate_likelihood(experiment.right, core.MODEL_FIT/'exp1_right_likelihood.pkl')
# model_fit.create_surrogate_likelihood(experiment, core.MODEL_FIT/'exp1_likelihood.pkl')

# model_fit.create_posterior_trace(core.MODEL_FIT/'exp1_left_likelihood.pkl', core.MODEL_FIT/'exp1_left_posterior.pkl')
# model_fit.create_posterior_trace(core.MODEL_FIT/'exp1_right_likelihood.pkl', core.MODEL_FIT/'exp1_right_posterior.pkl')
# model_fit.create_posterior_trace(core.MODEL_FIT/'exp1_likelihood.pkl', core.MODEL_FIT/'exp1_posterior.pkl')


##############################################################################
# Print parameter estimates and credible intervals
##############################################################################
# exp_analysis.print_posterior_summary(experiment)


##############################################################################
# Make the model fit figure for the manuscript
##############################################################################
# exp_analysis.make_posterior_projections_figure(experiment, core.FIGS/'posterior_projections.eps', max_normalize=True)


##############################################################################
# Plot posterior predictive checks
##############################################################################
# exp_analysis.plot_posterior_predictive_checks(experiment, core.FIGS/'posterior_predictive.eps')


##############################################################################
# Make the uncertainty prediction figure for the manuscript
##############################################################################
# exp_analysis.plot_uncertainty_prediction(experiment, core.MODEL_FIT/'exp1.pkl', core.FIGS/'uncertainty_prediction.eps')
