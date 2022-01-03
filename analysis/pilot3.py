'''

This script is the main entry point for the analysis of Experiment 2. To
reproduce the analyses, simply uncomment the particular lines below that you
are interested in. Then, if you want to dig into the details, look up the
relevant functions in exp_analysis.py

'''

import core
import exp_analysis


##############################################################################
# Load the experiment data and set the exclusion threshold
##############################################################################
experiment = exp_analysis.Experiment('pilot3')
experiment.set_exclusion_threshold(7, 8)

experiment.left.get_user('01').exclude() # random-x version
experiment.left.get_user('05').exclude() # calibration problems
experiment.right.get_user('01').exclude() # random-x version
experiment.right.get_user('02').exclude() # failure to learn


##############################################################################
# Print median completion time
##############################################################################
# exp_analysis.calculate_median_completion_time(experiment)


##############################################################################
# Make various plots
##############################################################################
# exp_analysis.make_all_trial_images(core.VISUALS, experiment)
# exp_analysis.plot_all_landing_distributions(core.VISUALS, experiment, separate_quick_and_slow=True)
exp_analysis.plot_overall_landing_distributions(core.VISUALS, experiment, show_individual_curves=True)


##############################################################################
# Make the results figure for the manuscript
##############################################################################
# exp_analysis.make_results_figure(experiment, core.FIGS/'exp1_results.eps')


##############################################################################
# Fit a new posterior, using the posterior from Experiment 1 as a prior.
##############################################################################
# import model_fit
# initial_evaluation = [0.88, 0.11, 0.33, 0.07]
# model_fit.create_surrogate_likelihood(experiment.left, core.MODEL_FIT/'exp2_left_likelihood.pkl', initial_evaluation)
# model_fit.create_surrogate_likelihood(experiment.right, core.MODEL_FIT/'exp2_right_likelihood.pkl', initial_evaluation)
# model_fit.create_surrogate_likelihood(experiment, core.MODEL_FIT/'exp2_likelihood.pkl', initial_evaluation)


##############################################################################
# Fit a new posterior, using the posterior from Experiment 1 as a prior.
##############################################################################
# import model_fit
# prior = model_fit.create_prior_from_posterior(core.MODEL_FIT/'exp1_posterior.pkl')
# model_fit.fit_posterior(prior, core.MODEL_FIT/'exp2_left_likelihood.pkl', core.MODEL_FIT/'exp2_left_posterior.pkl')
# model_fit.fit_posterior(prior, core.MODEL_FIT/'exp2_right_likelihood.pkl', core.MODEL_FIT/'exp2_right_posterior.pkl')
# model_fit.fit_posterior(prior, core.MODEL_FIT/'exp2_likelihood.pkl', core.MODEL_FIT/'exp2_posterior.pkl', n_samples=1000)


##############################################################################
# Print parameter estimates and credible intervals
##############################################################################
# exp_analysis.print_posterior_summary(experiment)


##############################################################################
# Make the model fit figure for the manuscript
##############################################################################
# exp_analysis.make_posterior_projections_figure(experiment, core.FIGS/'posterior_projections2.eps', show_each_condition=False)


##############################################################################
# Plot posterior predictive checks
##############################################################################
# exp_analysis.plot_posterior_predictive_checks(experiment, core.FIGS/'posterior_predictive.eps')


##############################################################################
# Make the uncertainty prediction figure for the manuscript
##############################################################################
# exp_analysis.plot_uncertainty_prediction(experiment, core.MODEL_FIT/'exp1.pkl', core.FIGS/'uncertainty_prediction.eps')


##############################################################################
# Plot landing position distribution
##############################################################################
# exp_analysis.plot_landing_distribution(experiment, core.VISUALS / 'landing_distribution.pdf')
