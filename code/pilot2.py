import core
import exp_analysis

visuals_dir = core.VISUALS / 'pilot2'
task_l = exp_analysis.Task('pilot2_left')
task_r = exp_analysis.Task('pilot2_right')

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