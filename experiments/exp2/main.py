'''
PsychoPy implementation of Experiment 2. This code was written for PsychoPy
version 2020.2.10, and is intended to be used with an EyeLink 1000 eye
tracker. To run the script for real, you first need to obtain PyLink and
EyeLinkCoreGraphicsPsychoPy. Alternatively, set TEST_MODE to True and use the
mouse to simulate the fixation position.

The experiment is run with a command like

    python main.py exp2_left 31

where exp2_left is a task ID and 31 is a participant ID. The script expects to
find task config files in the specified DATA_DIR.

To terminate the experiment, press the Q key (for quit) and the experiment
will exit after the current trial.

During eye tracking trials, you can force calibration by pressing the C key
(for calibrate), which will interrupt the current trial and return to it
after calibration.
'''


from psychopy import core, event, monitors, visual
from pathlib import Path
from time import time
import random
import json


DATA_DIR = Path('../../data/experiments/')

# screen metrics
SCREEN_WIDTH_PX = 1920
SCREEN_HEIGHT_PX = 1080
SCREEN_WIDTH_MM = 587
SCREEN_DISTANCE_MM = 570

# area of the screen that is actually used
PRESENTATION_WIDTH_PX = 960
PRESENTATION_HEIGHT_PX = 540

BUTTON_SIZE_PX = 100 # size of object buttons
FIXATION_TOLERANCE_PX = 18 # permissible distance from the fixation dot
TIME_RESOLUTION_SECONDS = 0.01 # time to wait between gaze position polls
FONT_WIDTH_TO_HEIGHT_RATIO = 1.66666667 # in Courier New, this ratio is 1 : 1 2/3

# for test purposes
TEST_MODE = False
SKIP_TRAINING = False
SKIP_FREE_FIXATION_TEST = False
SKIP_CONTROLLED_FIXATION_TEST = False


if not TEST_MODE:
    import pylink
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


class InterruptTrialForRecalibration(Exception):
    pass


class Experiment:

    def __init__(self, task_id, user_id):

        # Load task data
        self.task_data_path = DATA_DIR / f'{task_id}.json'
        with open(self.task_data_path) as file:
            self.task_data = json.load(file)

        # Calculate screen and font metrics
        px_per_mm = SCREEN_WIDTH_PX / SCREEN_WIDTH_MM
        self.char_width = int(round(self.task_data['char_width_mm'] * px_per_mm))
        self.char_height = self.char_width * FONT_WIDTH_TO_HEIGHT_RATIO
        self.word_width = self.char_width * self.task_data['n_letters']
        self.horizontal_margin = (SCREEN_WIDTH_PX - PRESENTATION_WIDTH_PX) // 2
        self.vertical_margin = (SCREEN_HEIGHT_PX - PRESENTATION_HEIGHT_PX) // 2

        # Calculate button positions
        first_button_position = self.horizontal_margin + BUTTON_SIZE_PX // 2
        inter_button_distance = (SCREEN_WIDTH_PX - self.horizontal_margin * 2 - BUTTON_SIZE_PX * self.task_data['n_items']) / (self.task_data['n_items'] - 1)
        button_positions = [
            self.transform_to_center_origin(
                x=int(round(first_button_position + button_i * inter_button_distance + button_i * BUTTON_SIZE_PX)),
                y=SCREEN_HEIGHT_PX - self.vertical_margin - BUTTON_SIZE_PX // 2
            ) for button_i in range(self.task_data['n_items'])
        ]
        button_rects = [
            (
                *self.transform_to_top_left_origin(x - BUTTON_SIZE_PX // 2, y + BUTTON_SIZE_PX // 2),
                BUTTON_SIZE_PX,
                BUTTON_SIZE_PX
            ) for x, y in button_positions
        ]

        # Load user_data or create new user
        self.user_data_path = DATA_DIR / task_id / f'{user_id}.json'
        if self.user_data_path.exists():
            with open(self.user_data_path) as file:
                self.user_data = json.load(file)
        else:
            self.user_data = {
                'user_id': user_id,
                'task_id': task_id,
                'creation_time': int(time()),
                'modified_time': None,
                'screen_width_px': SCREEN_WIDTH_PX,
                'screen_height_px': SCREEN_HEIGHT_PX,
                'screen_width_mm': SCREEN_WIDTH_MM,
                'screen_distance_mm': SCREEN_DISTANCE_MM,
                'presentation_area': [
                    self.horizontal_margin,
                    self.vertical_margin,
                    PRESENTATION_WIDTH_PX,
                    PRESENTATION_HEIGHT_PX,
                ],
                'buttons': button_rects,
                'font_size': self.char_height,
                'word_forms': generate_word_forms(self.task_data),
                'object_array': generate_object_array(self.task_data),
                'trial_sequence': generate_trial_sequence(self.task_data),
                'responses': {
                    'mini_test': [],
                    'free_fixation_test': [],
                    'controlled_fixation_test': [],
                },
                'sequence_position': 0,
            }
            self.save_user_data()

        # Convert millisecond times in task_data to seconds
        self.trial_time = self.task_data['trial_time'] / 1000
        self.pause_time = self.task_data['pause_time'] / 1000
        self.reveal_time = self.task_data['reveal_time'] / 1000
        self.gaze_time = self.task_data['gaze_time'] / 1000
        self.n_trials_until_calibration = 0

        # Set up monitor and window
        self.monitor = monitors.Monitor('monitor', width=SCREEN_WIDTH_MM, distance=SCREEN_DISTANCE_MM)
        self.monitor.setSizePix((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX))
        self.window = visual.Window((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), monitor=self.monitor, fullscr=not TEST_MODE, winType='pyglet', units='pix', allowStencil=True, color=(1, 1, 1))

        if not TEST_MODE:
            # Set up eye tracker connection
            self.tracker = pylink.EyeLink('100.1.1.1')
            self.tracker.openDataFile('ovp.edf')
            self.tracker.sendCommand("add_file_preamble_text 'OVP Experiment 2'")
            pylink.openGraphicsEx(EyeLinkCoreGraphicsPsychoPy(self.tracker, self.window))
            self.tracker.setOfflineMode()
            pylink.pumpDelay(100)
            self.tracker.sendCommand(f'screen_pixel_coords = 0 0 {SCREEN_WIDTH_PX-1} {SCREEN_HEIGHT_PX-1}')
            self.tracker.sendMessage(f'DISPLAY_COORDS = 0 0 {SCREEN_WIDTH_PX-1} {SCREEN_HEIGHT_PX-1}')
            self.tracker.sendCommand('sample_rate 1000')
            self.tracker.sendCommand('recording_parse_type = GAZE')
            self.tracker.sendCommand('select_parser_configuration 0')
            self.tracker.sendCommand('calibration_type = HV13')
            proportion_w = PRESENTATION_WIDTH_PX // SCREEN_WIDTH_PX
            proportion_h = PRESENTATION_HEIGHT_PX // SCREEN_HEIGHT_PX
            self.tracker.sendCommand(f'calibration_area_proportion = {proportion_w} {proportion_h}')
            self.tracker.sendCommand(f'validation_area_proportion = {proportion_w} {proportion_h}')
            self.tracker.sendCommand('file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT')
            self.tracker.sendCommand('file_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,INPUT')
            self.tracker.sendCommand('link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT')
            self.tracker.sendCommand('link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,PUPIL,HREF,AREA,STATUS,INPUT')

        # Set up mouse
        self.mouse = event.Mouse(visible=True, win=self.window)
        self.mouse.clickReset()

        # Create fixation dot
        self.fixation_dot = visual.Circle(self.window,
            lineColor='black',
            radius=SCREEN_WIDTH_PX / 256,
            lineWidth=SCREEN_WIDTH_PX / 256,
        )

        # Create word stimuli for use during training
        self.training_word_stims = [
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=(0, -200),
                height=self.char_height,
                text=word,
            ) for word in self.user_data['word_forms']
        ]

        # Create object stimuli for use during training
        stim_size = 40 * px_per_mm
        self.training_object_stims = [
            visual.ImageStim(self.window,
                image=Path('images') / 'objects' / f'{object_i}.png',
                pos=(0, 100),
                size=(stim_size, stim_size),
            ) for object_i in range(self.task_data['n_items'])
        ]

        # Create word stimuli for use in test trials
        self.test_word_stims = [
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=(0, 0),
                height=self.char_height,
                text=word,
            ) for word in self.user_data['word_forms']
        ]

        # Create object buttons for use in test trials
        self.object_buttons = [
            visual.ImageStim(self.window,
                image=Path('images') / 'object_buttons' / f'{object_i}.png',
                size=(BUTTON_SIZE_PX, BUTTON_SIZE_PX),
                pos=button_position,
            ) for object_i, button_position in zip(self.user_data['object_array'], button_positions)
        ]

    def save_user_data(self):
        '''
        Write the current state of the user_data dictionary to JSON.
        '''
        self.user_data['modified_time'] = int(time())
        with open(self.user_data_path, 'w') as file:
            json.dump(self.user_data, file, indent='\t')

    def save_response(self, trial_type, response_data):
        '''
        Store response data and save current state of user_data.
        '''
        response_data['time'] = int(time())
        self.user_data['responses'][trial_type].append(response_data)
        self.save_user_data()

    def save_screenshot(self, filename):
        '''
        Save a screenshot of the state of the current window (mostly used for
        testing purposes).
        '''
        image = self.window.getMovieFrame()
        image.save(filename)

    def transform_to_center_origin(self, x, y):
        '''
        Transform xy-coordinates based on a top-left origin into
        xy-coordinates based on a center origin.
        '''
        return x - SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX // 2 - y

    def transform_to_top_left_origin(self, x, y):
        '''
        Transform xy-coordinates based on a center origin into xy-coordinates
        based on a top-left origin.
        '''
        return x + SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX // 2 - y

    def get_gaze_position(self):
        '''
        Returns the current gaze position from the eye tracker (with a center
        origin). Before requesting the sample, a short pause is performed so
        as not to flood the eye tracker with requests. If in test mode, this
        returns the mouse position instead.
        '''
        core.wait(TIME_RESOLUTION_SECONDS)
        if TEST_MODE:
            return self.mouse.getPos()
        gaze_sample = self.tracker.getNewestSample()
        if gaze_sample.isRightSample():
            x, y = gaze_sample.getRightEye().getGaze()
        else:
            x, y = gaze_sample.getLeftEye().getGaze()
        return self.transform_to_center_origin(x, y)

    def perform_calibration(self):
        '''
        Run through the eye tracker calibration sequence. In test mode, this
        is skipped.
        '''
        visual.TextStim(self.window,
            color='black',
            text='Calibrazione... Mettiti comodo...',
        ).draw()
        self.window.flip()
        if not TEST_MODE:
            self.tracker.doTrackerSetup()
        self.n_trials_until_calibration = self.task_data['calibration_freq']

    def abandon_and_recalibrate(self):
        '''
        Abandon the current trial and perform calibration. After
        recalibration, we will return to the abandoned trial.
        '''
        self.tracker.sendMessage('trial_abandoned')
        self.tracker.stopRecording()
        self.perform_calibration()

    def show_response_buttons(self):
        '''
        Draw all the object buttons to the screen.
        '''
        for object_button in self.object_buttons:
            object_button.draw()
        self.window.flip()

    def show_feedback(self, correct_item):
        '''
        Give feedback to the participant by showing only the correct button.
        '''
        correct_button_i = self.user_data['object_array'].index(correct_item)
        self.object_buttons[correct_button_i].draw()
        self.window.flip()
        core.wait(self.pause_time * 2)

    def show_fixation_dot(self):
        '''
        Show the fixation dot in the center of the screen.
        '''
        self.fixation_dot.draw()
        self.window.flip()

    def await_mouse_selection(self):
        '''
        Wait for a mouse click and then check if the click is on one of the
        object buttons; if so, return the selected item.
        '''
        self.mouse.clickReset()
        while True:
            core.wait(TIME_RESOLUTION_SECONDS)
            if self.mouse.getPressed()[0]:
                mouse_position = self.mouse.getPos()
                for button_i, object_button in enumerate(self.object_buttons):
                    if object_button.contains(mouse_position):
                        return self.user_data['object_array'][button_i]

    def await_gaze_selection(self):
        '''
        Wait for the participant to fixate an object for the specified time
        and return the selected item. If the C key is pressed, the trial will
        be abandoned in order to recalibrate.
        '''
        fixated_button_i = None
        gaze_timer = core.Clock()
        while True:
            if event.getKeys(['c']):
                raise InterruptTrialForRecalibration
            gaze_position = self.get_gaze_position()
            for button_i, object_button in enumerate(self.object_buttons):
                if object_button.contains(gaze_position):
                    if button_i == fixated_button_i:
                        # still looking at the same button
                        if gaze_timer.getTime() >= self.gaze_time:
                            # and gaze has been on button for sufficient time
                            return self.user_data['object_array'][button_i]
                    else: # gaze has moved to different button, reset timer
                        fixated_button_i = button_i
                        gaze_timer.reset()
                    break
            else: # gaze is not on any button, reset timer
                fixated_button_i = None
                gaze_timer.reset()

    def await_fixation_on_fixation_dot(self):
        '''
        Wait for the participant to fixate the fixation dot for the specified
        time. If the C key is pressed, the trial will be abandoned in order
        to recalibrate.
        '''
        gaze_timer = core.Clock()
        while True:
            if event.getKeys(['c']):
                raise InterruptTrialForRecalibration
            x, y = self.get_gaze_position()
            distance_from_origin = (x ** 2 + y ** 2) ** 0.5
            if distance_from_origin < FIXATION_TOLERANCE_PX:
                if gaze_timer.getTime() >= self.gaze_time:
                    return True
            else:
                gaze_timer.reset()

    def await_word_entry(self, word_object):
        '''
        Wait for the participant's gaze to enter the word object. If the C key
        is pressed, the trial will be abandoned in order to recalibrate.
        '''
        while True:
            if event.getKeys(['c']):
                raise InterruptTrialForRecalibration
            if word_object.contains(self.get_gaze_position()):
                return True

    def instructions(self, image=None, message=None):
        '''
        Display an instructional image or message and await a press of the
        space bar to continue.
        '''
        if image:
            visual.ImageStim(self.window,
                image=Path(f'images/instructions/{image}'),
                size=(1000, 600),
            ).draw()
        elif message:
            visual.TextStim(self.window,
                color='black',
                text=message,
            ).draw()
        self.window.flip()
        event.waitKeys(keyList=['space'])
        self.n_trials_until_calibration = 0

    def training_block(self, training_items, test_item):
        '''
        Run a block of training (passive exposure trials, followed by a
        mini-test).
        '''
        if SKIP_TRAINING:
            return
        # display each passive exposure trial
        for training_item in training_items:
            self.training_object_stims[training_item].draw()
            self.window.flip()
            core.wait(self.pause_time)
            self.training_object_stims[training_item].draw()
            self.training_word_stims[training_item].draw()
            self.window.flip()
            core.wait(self.trial_time - self.pause_time)
            self.window.flip()
            core.wait(self.pause_time)
        # display the test item and response buttons
        self.test_word_stims[test_item].draw()
        self.show_response_buttons()
        # await response and show feedback
        selected_item = self.await_mouse_selection()
        self.test_word_stims[test_item].draw()
        self.show_feedback(test_item)
        # save response
        self.save_response('mini_test', {
            'target_item': test_item,
            'selected_item': selected_item,
        })

    def generate_random_word_position(self):
        '''
        Choose a random y-position within the presentation area of the screen,
        either above the fixation point or below the fixation point. The word
        must be at least one full line height away from the fixation point.
        '''
        if random.random() < 0.5:
            y1 = int(self.vertical_margin + self.char_height // 2)
            y2 = int(SCREEN_HEIGHT_PX // 2 - self.char_height * 2)
        else:
            y1 = int(SCREEN_HEIGHT_PX // 2 + self.char_height * 2)
            y2 = int(SCREEN_HEIGHT_PX - self.vertical_margin - self.char_height // 2)
        x = SCREEN_WIDTH_PX // 2
        y = random.randrange(y1, y2)
        return x, y

    def free_fixation_test(self, target_item):
        '''
        Run a free-fixation trial. Participant must fixate a dot for 3
        seconds, after which a word is flashed up real quick in a random
        position on the screen. The participant is then shown the object
        stimuli and must select one by holding their gaze on it for 3
        seconds.
        '''
        if SKIP_FREE_FIXATION_TEST:
            return
        if not TEST_MODE:
            self.mouse.setVisible(False)
        # determine position to place the word
        word_position_tl = self.generate_random_word_position()
        word_position = self.transform_to_center_origin(*word_position_tl)
        # potentially perform calibration
        if self.n_trials_until_calibration == 0:
            self.perform_calibration()
        self.n_trials_until_calibration -= 1
        # initialize eye tracker recording
        if not TEST_MODE:
            self.tracker.startRecording(1, 1, 1, 1)
            self.tracker.sendMessage('trial_type free_fixation_test')
            self.tracker.sendMessage(f'target_item {target_item}')
            self.tracker.sendMessage(f'word_position_x {word_position_tl[0]}')
            self.tracker.sendMessage(f'word_position_y {word_position_tl[1]}')
        # show fixation dot and await fixation
        self.show_fixation_dot()
        self.await_fixation_on_fixation_dot()
        # show word stim
        self.test_word_stims[target_item].pos = word_position
        self.test_word_stims[target_item].draw()
        if not TEST_MODE:
            self.tracker.sendMessage('start_word_presentation')
        self.window.flip()
        # await entry into the word boundary
        self.await_word_entry(self.test_word_stims[target_item])
        # hide word after reveal time
        core.wait(self.reveal_time)
        self.window.flip()
        if not TEST_MODE:
            self.tracker.sendMessage('end_word_presentation')
        core.wait(self.pause_time)
        # show response buttons, await response, and show feedback
        self.show_response_buttons()
        selected_item = self.await_gaze_selection()
        self.show_feedback(target_item)
        # stop recording and save response
        if not TEST_MODE:
            self.tracker.stopRecording()
        self.save_response('free_fixation_test', {
            'target_item': target_item,
            'word_position': word_position_tl,
            'selected_item': selected_item,
        })

    def calculate_eccentricity(self, fixation_position):
        '''
        Calculate how many pixels left or right the word must be shifted to
        bring a particular letter position into the center.
        '''
        eccentricity = (int(self.task_data['n_letters'] / 2) - fixation_position) * self.char_width
        if self.task_data['n_letters'] % 2 == 0:
            eccentricity -= self.char_width // 2
        return eccentricity

    def controlled_fixation_test(self, target_item, fixation_position):
        '''
        Run a controlled-fixation trial. Participant must fixate a dot for 3
        seconds, after which a word is flashed up real quick at a particular
        controlled position in the word. The participant is then shown the
        object stimuli and must select one by holding their gaze on it for 3
        seconds.
        '''
        if SKIP_CONTROLLED_FIXATION_TEST:
            return
        if not TEST_MODE:
            self.mouse.setVisible(False)
        # determine position to place the word
        eccentricity = self.calculate_eccentricity(fixation_position)
        word_position = (eccentricity, 0)
        word_position_tl = self.transform_to_top_left_origin(*word_position)
        # potentially perform calibration
        if self.n_trials_until_calibration == 0:
            self.perform_calibration()
        self.n_trials_until_calibration -= 1
        # initialize eye tracker recording
        if not TEST_MODE:
            self.tracker.startRecording(1, 1, 1, 1)
            self.tracker.sendMessage('trial_type controlled_fixation_test')
            self.tracker.sendMessage(f'target_item {target_item}')
            self.tracker.sendMessage(f'word_position_x {word_position_tl[0]}')
            self.tracker.sendMessage(f'word_position_y {word_position_tl[1]}')
        # show fixation dot and await fixation
        self.show_fixation_dot()
        self.await_fixation_on_fixation_dot()
        # show word stim
        self.test_word_stims[target_item].pos = word_position
        self.test_word_stims[target_item].draw()
        if not TEST_MODE:
            self.tracker.sendMessage('start_word_presentation')
        self.window.flip()
        # hide word after reveal time
        core.wait(self.reveal_time)
        self.window.flip()
        if not TEST_MODE:
            self.tracker.sendMessage('end_word_presentation')
        core.wait(self.pause_time)
        # show response buttons, await response, and show feedback
        self.show_response_buttons()
        selected_item = self.await_gaze_selection()
        self.show_feedback(target_item)
        # stop recording and save response
        if not TEST_MODE:
            self.tracker.stopRecording()
        self.save_response('controlled_fixation_test', {
            'target_item': target_item,
            'word_position': word_position_tl,
            'fixation_position': fixation_position,
            'selected_item': selected_item,
        })

    def execute(self):
        '''
        Execute the experiment: Iterate over the trial sequence and run each
        trial. If the Q key has been pressed, terminate the experiment after
        the current trial. If a trial completes successfully, the sequence
        position is incremented and the current user_data is saved.
        '''
        while self.user_data['sequence_position'] < len(self.user_data['trial_sequence']):
            if event.getKeys(['q']):
                break
            trial_type, params = self.user_data['trial_sequence'][self.user_data['sequence_position']]
            trial_func = getattr(self, trial_type)
            try:
                trial_func(**params)
            except InterruptTrialForRecalibration:
                self.abandon_and_recalibrate()
            else:
                self.user_data['sequence_position'] += 1
        visual.TextStim(self.window,
            color='black',
            text='Esperimento completato',
        ).draw()
        self.window.flip()
        if not TEST_MODE:
            self.tracker.setOfflineMode()
            pylink.pumpDelay(100)
            self.tracker.closeDataFile()
            pylink.pumpDelay(500)
            edf_data_path = DATA_DIR / self.user_data['task_id'] / f'{self.user_data["user_id"]}.edf'
            suffix = 1
            while edf_data_path.exists():
                edf_data_path = DATA_DIR / self.user_data['task_id'] / f'{self.user_data["user_id"]}_{suffix}.edf'
                suffix += 1
            self.tracker.receiveDataFile('ovp.edf', str(edf_data_path))
            self.tracker.close()
        event.waitKeys()
        core.quit()


def generate_word_forms(task):
    '''
    Generates a set of word forms by first shuffling the alphabet. Each
    participant sees a different set of surface forms, although the
    underlying structure is always the same (within condition).
    '''
    alphabet = []
    for symbols in task['alphabet']:
        random.shuffle(symbols)
        alphabet.extend(symbols)
    return [''.join([alphabet[letter_index] for letter_index in word]) for word in task['words']]

def generate_object_array(task):
    '''
    Generates a random order in which to present the object stimuli in the
    array.
    '''
    objects = list(range(task['n_items']))
    random.shuffle(objects)
    return objects

def generate_trial_sequence(task):
    '''
    Generates the main sequence of trials, including the training trials
    (passive exposure and mini-test) and both types of test.
    '''
    item_indices = list(range(task['n_items']))
    seen_items = []
    # TRAINING INSTRUCTIONS
    trial_sequence = [('instructions', {
        'image': 'training_it.png',
    })]
    # TRAINING TRIALS
    for i in range(task['training_reps']):
        for j in range(task['mini_test_freq']):
            training_items = []
            random.shuffle(item_indices)
            for k in range(task['n_items']):
                training_item = item_indices[k]
                if j == 0:
                    seen_items.append(training_item)
                training_items.append(training_item)
                if len(training_items) == task['mini_test_freq']:
                    random.shuffle(seen_items)
                    test_item = seen_items.pop()
                    trial_sequence.append(('training_block', {
                        'training_items': training_items,
                        'test_item': test_item,
                    }))
                    training_items = []
    # TEST INSTRUCTIONS
    trial_sequence.append(('instructions', {
        'image': 'test_it.png',
    }))
    # FREE-FIXATION TEST TRIALS
    test_trials = []
    for i in range(task['free_fixation_reps']):
        for j in range(task['n_items']):
            test_trials.append(('free_fixation_test', {
                'target_item': j,
            }))
    random.shuffle(test_trials)
    trial_sequence.extend(test_trials)
    # CONTROLLED-FIXATION TEST INSTRUCTIONS
    trial_sequence.append(('instructions', {
        'message': 'Le parole ora appariranno a sinistra o a destra del punto di fissazione anziché sopra o sotto',
    }))
    # CONTROLLED-FIXATION TEST TRIALS
    for i in range(task['controlled_fixation_reps']):
        test_trials = []
        for j in range(task['n_items']):
            for fixation_position in range(task['n_letters']):
                test_trials.append(('controlled_fixation_test', {
                    'target_item': j,
                    'fixation_position': fixation_position,
                }))
        random.shuffle(test_trials)
        trial_sequence.extend(test_trials)
    return trial_sequence


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', action='store', type=str, help='task ID')
    parser.add_argument('user_id', action='store', type=str, help='user ID')
    args = parser.parse_args()

    Experiment(args.task_id, args.user_id).execute()
