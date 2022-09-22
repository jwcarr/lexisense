from collections import defaultdict
from pathlib import Path
import numpy as np
import eyekit


ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / 'data'
EXP_DATA = DATA / 'experiments'
MODEL_FIT = DATA / 'model_fit'


class Participant:

	def __init__(self, ID, task_id):
		self.ID = ID
		self.task_id = task_id
		self._user_data = eyekit.io.load(EXP_DATA / self.task_id / f'{self.ID}.json')
		self.excluded = False

		# If particpant's responses are stored as flat lists, separate into test types
		if isinstance(self['responses'], list):
			self.trials = defaultdict(list)
			for response in self['responses']:
				response['target_item'] = response['object']
				response['selected_item'] = response['selected_object']
				self.trials[response['test_type']].append(response)
		else:
			self.trials = self['responses']
		
		# If participant has free fixation trials, adjust padding of the TextBlock objects
		for trial in self.iter_free_fixation_trials():
			### UNCLEAR HOW THE PADDING SHOULD BE SET
			# trial['word'][0:0:7].adjust_padding(bottom=-10) # more restrictive
			trial['word'][0:0:7].adjust_padding(top=10) # less restrictive
			# trial['word'][0:0:7].adjust_padding(top=6, bottom=-3) # symmetric within PsychoPy box
			### UNCLEAR HOW THE PADDING SHOULD BE SET

	def __getitem__(self, key):
		return self._user_data[key]

	def unpack(self):
		return ((self,),)

	def exclude(self):
		self.excluded = True

	def iter_training_trials(self):
		for trial in self.trials['mini_test']:
			yield trial

	def iter_controlled_fixation_trials(self):
		for trial in self.trials['controlled_fixation_test']:
			yield trial

	def iter_free_fixation_trials(self, word_position=None):
		for trial in self.trials['free_fixation_test']:
			if word_position:
				if trial['word'].y > 540 and word_position == 'above':
					continue
				elif trial['word'].y < 540 and word_position == 'below':
					continue
			yield trial

	def iter_all_test_trials(self):
		for trial in self.iter_controlled_fixation_trials():
			yield trial
		for trial in self.iter_free_fixation_trials():
			yield trial

	def completion_time(self, use_first_trial_time=False):
		end_time = self['modified_time']
		if use_first_trial_time:
			start_time = self['responses'][0]['time'] + 60
		else:
			start_time = self['creation_time']		
		return end_time - start_time

	def learning_score(self, n_last_trials=8):
		return sum([trial['target_item'] == trial['selected_item'] for trial in self.trials['mini_test'][-n_last_trials:]])

	def ovp_score(self):
		return sum([trial['target_item'] == trial['selected_item'] for trial in self.trials['controlled_fixation_test']])

	def learning_curve(self, n_previous_trials=8):
		correct = [trial['target_item'] == trial['selected_item'] for trial in self.iter_training_trials()]
		return np.array([
			sum(correct[i-(n_previous_trials-1) : i+1]) for i in range(n_previous_trials-1, len(correct))
		]) / n_previous_trials

	def ovp_curve(self):
		n_successes_by_position = defaultdict(int)
		n_trials_by_position = defaultdict(int)
		for trial in self.iter_controlled_fixation_trials():
			position = trial['fixation_position']
			n_successes_by_position[position] += trial['target_item'] == trial['selected_item']
			n_trials_by_position[position] += 1
		n_successes_by_position = np.array([
			n_successes_by_position[i] for i in range(len(n_successes_by_position))
		])
		n_trials_by_position = np.array([
			n_trials_by_position[i] for i in range(len(n_trials_by_position))
		])
		return n_successes_by_position / n_trials_by_position

	def landing_positions(self, word_position=None):
		positions = []
		for trial in self.iter_free_fixation_trials(word_position):
			seq = trial['fixations']
			word_ia = trial['word'][0:0:7]
			for fixation in seq:
				if fixation.start > trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:
					if fixation in word_ia:
						px_position = int(fixation.x - word_ia.onset)
						positions.append(px_position)
						break
		return positions

	def landing_times(self, word_position=None):
		times = []
		for trial in self.iter_free_fixation_trials(word_position):
			seq = trial['fixations']
			word_ia = trial['word'][0:0:7]
			for fixation in seq:
				if fixation.start > trial['start_word_presentation'] and fixation.start < trial['end_word_presentation']:
					if fixation in word_ia:
						saccade_time = fixation.start - trial['start_word_presentation']
						if saccade_time < 500:
							times.append(saccade_time)
						break
		return times

class Task:

	def __init__(self, ID, colors=('black', 'gray')):
		self.ID = ID
		self.colors = colors
		self.params = {}
		self.priors = {}
		self.min_learning_score = 0
		self.n_last_trials = 8
		self.posterior_trace = None

	@property
	def n_trained_participants(self):
		return len([user for user in self.iter_with_excludes()])

	@property
	def n_retained_participants(self):
		return len([user for user in self])

	@property
	def color(self):
		return self.colors[0]

	@property
	def light_color(self):
		return self.colors[1]

	@property
	def likelihood_file(self):
		return MODEL_FIT / f'{self.ID}_likelihood.pkl'

	@property
	def posterior_file(self):
		return MODEL_FIT / f'{self.ID}_posterior.nc'

	def get_posterior(self, posterior_file=None):
		import arviz
		if posterior_file is not None:
			return arviz.from_netcdf(posterior_file)
		if self.posterior_trace is None:
			self.posterior_trace = arviz.from_netcdf(self.posterior_file)
		return self.posterior_trace

	def print_comments(self):
		for participant in self.iter_with_excludes():
			comments = participant['comments'].strip().replace('\n', ' ')
			print(f"{participant.task_id}_{participant.ID}: {comments}\n")

	def print_median_completion_time(self, use_first_trial_time=False):
		'''
		Calculate median completion time. Because participants often do not start the
		task immediately after accepting it, we will define this as the time between
		the first trial response and the final submission time, plus 60 seconds to
		account for initial instruction reading etc.
		'''
		times = []
		for participant in self.iter_with_excludes():
			times.append(participant.completion_time(use_first_trial_time))
		median_time = round(np.median(times) / 60)
		print(f'Median completion time of {median_time} minutes')

	def print_median_bonus(self):
		'''
		Calculate median bonus amount.
		'''
		bonuses = []
		for participant in self.iter_with_excludes():
			bonuses.append(participant['total_bonus'])
		median_bonus = round(np.median(bonuses))
		print(f'Median bonus: {median_bonus}')

	def print_n_exclusion_stats(self, test_type='free_fixation_test'):
		n_trials_by_status = {'complete':[], 'incomplete':[]}
		total_trials_run = 0
		for participant in self:
			n_trials = len(participant.trials['free_fixation_test'])
			total_trials_run += n_trials
			n_valid_trials = len(participant.landing_positions())
			if n_trials == 64:
				n_trials_by_status['complete'].append(n_valid_trials)
			else:
				n_trials_by_status['incomplete'].append(n_valid_trials)
		n_complete_datasets = len(n_trials_by_status['complete'])
		n_incomplete_datasets = len(n_trials_by_status['incomplete'])

		print(self.ID)
		print(f'{self.n_trained_participants} participants completed training')
		print(f'{self.n_retained_participants} participants remain after training exclusions')
		if n_complete_datasets:
			mean_dataset_size = round(np.mean(n_trials_by_status['complete']), 1)
			print(f'- {n_complete_datasets} complete datasets (on average {mean_dataset_size} trials were valid)')
		if n_incomplete_datasets:
			mean_dataset_size = round(np.mean(n_trials_by_status['incomplete']), 1)
			print(f'- {n_incomplete_datasets} incomplete datasets (on average {mean_dataset_size} trials were valid)')
		total_valid_trials = sum(n_trials_by_status['complete']) + sum(n_trials_by_status['incomplete'])
		print(f'Total valid trials: {total_valid_trials} ({round(total_valid_trials / total_trials_run * 100, 3)})')

	def print_test_accuracy(self):
		for condition in self.unpack():
			accuracy_scores = []
			for participant in condition:
				n_correct = 0
				n_trials = 0
				for trial in participant.iter_all_test_trials():
					if trial['target_item'] == trial['selected_item']:
						n_correct += 1
					n_trials +=1
				accuracy_scores.append(n_correct / n_trials)
			median = np.median(accuracy_scores)
			lo, hi = np.percentile(accuracy_scores, [25, 75])
			print(f'{condition.ID}: median = {median}, IQR = {lo} -- {hi}')


class Condition(Task):

	def __init__(self, ID, colors=('black', 'gray')):
		super(Condition, self).__init__(ID, colors)
		self._task_data = eyekit.io.load(EXP_DATA / f'{self.ID}.json')
		self._participants = []
		for participant_id in range(1, self['n_participants'] + 1):
			try:
				participant = Participant(str(participant_id).zfill(2), task_id=self.ID)
			except FileNotFoundError:
				print(f'Missing participant data file: {self.ID}, {participant_id}')
				continue
			self._participants.append(participant)

	@property
	def lexicon(self):
		return list(map(tuple, self['words']))

	def __getitem__(self, key):
		return self._task_data[key]

	def __iter__(self):
		for participant in self._participants:
			if participant.excluded:
				continue
			if participant.learning_score(self.n_last_trials) >= self.min_learning_score:
				yield participant

	def iter_with_excludes(self):
		for participant in self._participants:
			yield participant

	def get_participant(self, participant_id):
		for participant in self.iter_with_excludes():
			if participant.ID == participant_id:
				return participant
		return None

	def unpack(self):
		return (self,)

	def set_exclusion_threshold(self, min_learning_score, n_last_trials):
		self.min_learning_score = min_learning_score
		self.n_last_trials = n_last_trials

	def set_params(self, params):
		self.params |= params

	def set_priors(self, priors):
		self.priors |= priors

	def get_CFT_dataset(self):
		'''
		Convert individual task results to dataset format for fitting to the model.
		'''
		dataset = []
		for participant in self:
			for trial in participant.iter_controlled_fixation_trials():
				t = trial['target_item']
				j = trial['fixation_position']
				w = trial['selected_item']
				dataset.append((0, t, j, w))
		return dataset, [self.lexicon]

	def get_FFT_dataset(self, word_position=None):
		'''
		Return dataset of landing positions suitable for fitting the landing model.
		'''
		landing_x = []
		subject_idx = []
		participant_i = 0
		for participant in self:
			positions = participant.landing_positions(word_position)
			landing_x.extend(positions)
			subject_idx.extend([participant_i] * len(positions))
			participant_i += 1
		return landing_x, subject_idx


class Experiment(Task):

	def __init__(self, ID, colors=('black', 'gray')):
		super(Experiment, self).__init__(ID, colors)
		self.left = Condition(f'{ID}_left', ('cadetblue', '#AFD0D0'))
		self.right = Condition(f'{ID}_right', ('crimson', '#F58CA1'))

	def __iter__(self):
		for participant in self.left:
			yield participant
		for participant in self.right:
			yield participant

	def iter_with_excludes(self):
		for participant in self.left.iter_with_excludes():
			yield participant
		for participant in self.right.iter_with_excludes():
			yield participant

	def unpack(self):
		return (self.left, self.right)

	def set_exclusion_threshold(self, min_learning_score, n_last_trials):
		self.min_learning_score = min_learning_score
		self.n_last_trials = n_last_trials
		self.left.set_exclusion_threshold(min_learning_score, n_last_trials)
		self.right.set_exclusion_threshold(min_learning_score, n_last_trials)

	def set_params(self, params):
		self.params |= params
		self.left.set_params(params)
		self.right.set_params(params)

	def set_priors(self, priors):
		self.priors |= priors
		self.left.set_priors(priors)
		self.right.set_priors(priors)

	def get_CFT_dataset(self):
		'''
		Convert entire experimental results to dataset format for fitting to
		the model.
		'''
		dataset = []
		lexicons = []
		for l, condition in enumerate(self.unpack()):
			lexicons.append(condition.lexicon)
			for participant in condition:
				for trial in participant.iter_controlled_fixation_trials():
					t = trial['target_item']
					j = trial['fixation_position']
					w = trial['selected_item']
					dataset.append((l, t, j, w))
		return dataset, lexicons
