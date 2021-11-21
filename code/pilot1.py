import core
import exp_analysis
import model_fit

visuals_dir = core.VISUALS / 'pilot1'
task_l = exp_analysis.Task('pilot1_left')
task_r = exp_analysis.Task('pilot1_right')

tasks = [task_l, task_r]


# exp_analysis.print_comments(tasks)

# exp_analysis.calculate_median_completion_time(tasks)

# exp_analysis.check_size_selections(tasks)

# exp_analysis.plot_individual_results(visuals_dir, tasks)

# exp_analysis.plot_learning_scores(visuals_dir, tasks)

# exp_analysis.plot_learning_curves(visuals_dir, tasks)

# exp_analysis.plot_ovp_curves(visuals_dir, tasks, min_learning_score=7, show_individual_curves=True)

# exp_analysis.plot_training_inferences(visuals_dir, tasks)

# exp_analysis.plot_test_inferences(visuals_dir, tasks)




# STEP 1: fit the model to the data
# model_fit.fit_model(tasks, 'model_4', min_learning_score=7)

# STEP 2: view the model fit results
# model_fit.view_model_fit(core.DATA / 'model_fit' / 'model_4.pkl')

# best_fit_values = model_fit.get_model_fit(core.DATA / 'model_fit' / 'model_2.pkl')
# print(best_fit_values)


# STEP 3: generate a synthetic dataset based on fit parameters
# generate_synthetic_dataset_from_model_fit('exp1_model1', conditions, [0.7091, 0.1, 0.1473, 0.0603])
# generate_synthetic_dataset_from_model_fit('exp1_model2', conditions, [0.5542, 0.9999, 0.1, 0.0382, 0.0606])
# generate_synthetic_dataset_from_model_fit('exp1_model3', conditions, [0.6849, 0.8839, 0.3738, 0.01, 0.0987, 0.0412])

# STEP 4: plot the OVP curves and word inferences of the best fit model against the experimental data
# plot_ovp_curves(conditions[0], conditions[1], 30, 7, False, 'exp1_model1', 'crimson')
# plot_word_inferences(conditions, 30, 7, 'exp1_model1', 'crimson')

# plot_ovp_curves(conditions[0], conditions[1], 30, 7, False, 'exp1_model2', 'cadetblue')
# plot_word_inferences(conditions, 30, 7, 'exp1_model2', 'cadetblue')

# plot_ovp_curves(conditions[0], conditions[1], 30, 7, False, 'exp1_model3', 'cadetblue')	
# plot_word_inferences(conditions, 30, 7, 'exp1_model3', 'cadetblue')
