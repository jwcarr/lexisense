'''
This code was written for PsychoPy version 2020.2.10, and is intended to be
used with an EyeLink 1000 eye tracker. To run the script for real, you first
need to obtain PyLink and EyeLinkCoreGraphicsPsychoPy. Alternatively, set
TEST_MODE to True and use the mouse to simulate the gaze position.

The experiment is run with a command like

    python main.py exp1 01

where exp1 is a task ID and 01 is a participant ID. The script expects to
find task config files in the specified DATA_DIR.

To terminate the experiment, press the Q key (for quit) and the experiment
will exit once the current trial has been completed.

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

LANGUAGE = 'it' # language used for instructions, 'en' or 'it'

TEST_MODE = True # if set to True, use mouse to simulate gaze position
SKIP_TRAINING = True # if set to True, skip the training phase and go straight to the test

INSTRUCTION_CALIBRATION = {
    'en': 'New calibration... Get comfortable...',
    'it': 'Nuova calibrazione... Mettiti comodo...',
}

INSTRUCTION_END = {
    'en': 'Experiment complete...',
    'it': 'Esperimento completato...',
}

if not TEST_MODE:
    import pylink
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


class InterruptTrialAndRecalibrate(Exception):
    pass


class InterruptTrialAndExit(Exception):
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
        self.button_rects = [
            (
                *self.transform_to_top_left_origin(x - BUTTON_SIZE_PX // 2, y + BUTTON_SIZE_PX // 2),
                BUTTON_SIZE_PX,
                BUTTON_SIZE_PX
            ) for x, y in button_positions
        ]

        self.include_predictors = self.task_data['predictor_length'] > 0 and bool(self.task_data['predictor_forms'])
        self.predictor_tests = self.task_data['predictor_tests']

        # Load user_data or create new user
        self.user_data_path = DATA_DIR / task_id / f'{user_id}.json'
        if self.user_data_path.exists():
            with open(self.user_data_path) as file:
                self.user_data = json.load(file)
        else:
            if not self.user_data_path.parent.exists():
                self.user_data_path.parent.mkdir()
            if self.include_predictors:
                random.shuffle(self.task_data['predictor_forms'])
            object_mapping = list(range(self.task_data['n_items']))
            random.shuffle(object_mapping)
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
                'buttons': self.button_rects,
                'font_size': self.char_height,
                'predictor_mapping': self.task_data['predictor_forms'],
                'object_mapping': object_mapping,
                'phrase_forms': generate_phrase_forms(self.task_data),
                'predictor_array': generate_predictor_array(self.task_data),
                'object_array': generate_object_array(self.task_data),
                'trial_sequence': generate_trial_sequence(self.task_data),
                'responses': {
                    'mini_test': [],
                    'free_fixation_test': [],
                    'free_continuation_test': [],
                },
                'sequence_position': 0,
            }
            self.save_user_data()

        # Calculate word position metrics (with or without a predictor)
        if self.include_predictors:
            self.pred_word_width = self.char_width * self.task_data['predictor_length']
            self.targ_word_width = self.char_width * self.task_data['word_length']
            self.pred_position_test = (0, 0)
            self.targ_position_test = (self.pred_word_width // 2 + self.char_width + self.targ_word_width // 2, 0)
            offset = int((self.task_data['word_length'] + 1) / 2 * self.char_width)
            self.pred_position_trng = (self.pred_position_test[0] - offset, -200)
            self.targ_position_trng = (self.targ_position_test[0] - offset, -200)
            self.pred_position_mini = (self.pred_position_trng[0], 0)
            self.targ_position_mini = (self.targ_position_trng[0], 0)
        else:
            self.pred_word_width = None
            self.targ_word_width = self.char_width * self.task_data['word_length']
            self.pred_position_test = None
            self.targ_position_test = (0, 0)
            self.pred_position_trng = None
            self.targ_position_trng = (0, -200)
            self.pred_position_mini = None
            self.targ_position_mini = (0, 0)

        # Convert millisecond times in task_data to seconds
        self.trial_time = self.task_data['trial_time'] / 1000
        self.pause_time = self.task_data['pause_time'] / 1000
        self.reveal_time = self.task_data['reveal_time'] / 1000
        self.gaze_time = self.task_data['gaze_time'] / 1000
        self.boundary_x = (self.task_data['predictor_length'] + 1) / 2 * self.char_width
        self.n_trials_until_calibration = 0
        self.n_completed_trials = 0

        # Set up monitor and window
        self.monitor = monitors.Monitor('monitor', width=SCREEN_WIDTH_MM, distance=SCREEN_DISTANCE_MM)
        self.monitor.setSizePix((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX))
        self.window = visual.Window((SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX), monitor=self.monitor, fullscr=not TEST_MODE, winType='pyglet', units='pix', allowStencil=True, color=(1, 1, 1))

        if not TEST_MODE:
            # Set up eye tracker connection
            self.tracker = pylink.EyeLink('100.1.1.1')
            self.tracker.openDataFile('exp.edf')
            self.tracker.sendCommand("add_file_preamble_text 'Experiment 1'")
            pylink.openGraphicsEx(EyeLinkCoreGraphicsPsychoPy(self.tracker, self.window))
            self.tracker.setOfflineMode()
            pylink.pumpDelay(100)
            self.tracker.sendCommand(f'screen_pixel_coords = 0 0 {SCREEN_WIDTH_PX-1} {SCREEN_HEIGHT_PX-1}')
            self.tracker.sendMessage(f'DISPLAY_COORDS = 0 0 {SCREEN_WIDTH_PX-1} {SCREEN_HEIGHT_PX-1}')
            self.tracker.sendCommand('sample_rate 1000')
            self.tracker.sendCommand('recording_parse_type = GAZE')
            self.tracker.sendCommand('select_parser_configuration 0')
            self.tracker.sendCommand('calibration_type = HV13')
            proportion_w = PRESENTATION_WIDTH_PX / SCREEN_WIDTH_PX
            proportion_h = PRESENTATION_HEIGHT_PX / SCREEN_HEIGHT_PX
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

        # Create object stimuli for use during training
        stim_size = 40 * px_per_mm
        self.training_object_stims = [
            visual.ImageStim(self.window,
                image=Path('images') / 'objects' / f'{self.user_data["object_mapping"][object_i]}.png',
                pos=(0, 100),
                size=(stim_size, stim_size),
            ) for object_i in range(self.task_data['n_items'])
        ]

        # Create object buttons for use in test trials
        self.object_buttons = [
            visual.ImageStim(self.window,
                image=Path('images') / 'object_buttons' / f'{self.user_data["object_mapping"][object_i]}.png',
                size=(BUTTON_SIZE_PX, BUTTON_SIZE_PX),
                pos=button_position,
            ) for object_i, button_position in zip(self.user_data['object_array'], button_positions)
        ]

        self.training_phrase_stims = [(
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.pred_position_trng,
                height=self.char_height,
                text=pred,
            ),
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.targ_position_trng,
                height=self.char_height,
                text=targ,
            )) for pred, targ in self.user_data['phrase_forms']
        ]

        self.mini_test_phrase_stims = [(
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.pred_position_mini,
                height=self.char_height,
                text=pred,
            ),
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.targ_position_mini,
                height=self.char_height,
                text=targ,
            )) for pred, targ in self.user_data['phrase_forms']
        ]

        self.test_phrase_stims = [(
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.pred_position_test,
                height=self.char_height,
                text=pred,
            ),
            visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.targ_position_test,
                height=self.char_height,
                text=targ,
            )) for pred, targ in self.user_data['phrase_forms']
        ]

        # Create predictor buttons for use in predictor test trials
        if self.include_predictors:
            n_predictors = len(self.user_data['predictor_array'])
            s = self.task_data['n_items'] // 2 - n_predictors // 2
            e = s + n_predictors
            self.predictor_buttons = [
                visual.ImageStim(self.window,
                    image=Path('images') / 'predictor_button.png',
                    size=(BUTTON_SIZE_PX, BUTTON_SIZE_PX // 2),
                    pos=(button_position[0], button_position[1] + BUTTON_SIZE_PX // 2),
                ) for predictor_i, button_position in zip(self.user_data['predictor_array'], button_positions[s:e])
            ]
            self.predictor_button_labels = [
                visual.TextStim(self.window,
                    color='black',
                    font='Courier New',
                    pos=(button_position[0], button_position[1] + BUTTON_SIZE_PX // 2),
                    height=self.char_height // 2,
                    text=self.user_data["predictor_mapping"][predictor_i],
                ) for predictor_i, button_position in zip(self.user_data['predictor_array'], button_positions[s:e])
            ]
            self.missing_predictor_stim = visual.TextStim(self.window,
                color='black',
                font='Courier New',
                pos=self.pred_position_mini,
                height=self.char_height,
                text='?' * self.task_data['predictor_length'],
            )

    def save_user_data(self):
        '''
        Write the current state of the user_data dictionary to JSON.
        '''
        self.user_data['modified_time'] = int(time())
        with open(self.user_data_path, 'w') as file:
            json.dump(self.user_data, file, indent='\t')

    def store_response(self, trial_type, response_data):
        '''
        Store response data and save current state of user_data.
        '''
        response_data['time'] = int(time())
        self.user_data['responses'][trial_type].append(response_data)

    def save_screenshot(self, filename):
        '''
        Save a screenshot of the state of the current window (mostly used for
        testing purposes).
        '''
        image = self.window.getMovieFrame()
        image.save(filename)

    def save_tracker_recording(self, convert_to_asc=False):
        '''
        Save the eye tracker recording and close the connection. Ensure that
        the recording does not overwrite a file that already exists.
        '''
        if TEST_MODE:
            return
        self.tracker.setOfflineMode()
        pylink.pumpDelay(100)
        self.tracker.closeDataFile()
        pylink.pumpDelay(500)
        edf_data_path = DATA_DIR / self.user_data['task_id'] / f'{self.user_data["user_id"]}.edf'
        suffix = 1
        while edf_data_path.exists():
            edf_data_path = DATA_DIR / self.user_data['task_id'] / f'{self.user_data["user_id"]}_{suffix}.edf'
            suffix += 1
        self.tracker.receiveDataFile('exp.edf', str(edf_data_path))
        self.tracker.close()
        if convert_to_asc:
            from os import system
            system(f'edf2asc {edf_data_path}')

    def transform_to_center_origin(self, x, y):
        '''
        Transform xy-coordinates based on a top-left origin into
        xy-coordinates based on a center origin.
        '''
        return int(x - SCREEN_WIDTH_PX // 2), int(SCREEN_HEIGHT_PX // 2 - y)

    def transform_to_top_left_origin(self, x, y):
        '''
        Transform xy-coordinates based on a center origin into xy-coordinates
        based on a top-left origin.
        '''
        return int(x + SCREEN_WIDTH_PX // 2), int(SCREEN_HEIGHT_PX // 2 - y)

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
            text=INSTRUCTION_CALIBRATION[LANGUAGE],
        ).draw()
        self.window.flip()
        if not TEST_MODE:
            self.tracker.doTrackerSetup()
        self.n_trials_until_calibration = self.task_data['calibration_freq']

    def abandon_trial(self):
        '''
        Abandon the current trial. This stops eye tracker recording and writes
        a trial_abandoned message.
        '''
        if TEST_MODE:
            return
        self.tracker.sendMessage('trial_abandoned')
        self.tracker.stopRecording()

    def render_experimenter_screen(self, stims=[]):
        '''
        Render an outline of the screen on the host computer. In test mode,
        this is skipped.
        '''
        if TEST_MODE:
            return
        self.tracker.clearScreen(color=0)
        self.tracker.drawLine(
            (SCREEN_WIDTH_PX // 2, 0),
            (SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX),
            color=1
        )
        self.tracker.drawLine(
            (0, SCREEN_HEIGHT_PX // 2),
            (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX // 2),
            color=1
        )
        self.tracker.drawBox(
            SCREEN_WIDTH_PX // 2 - FIXATION_TOLERANCE_PX,
            SCREEN_HEIGHT_PX // 2 - FIXATION_TOLERANCE_PX,
            FIXATION_TOLERANCE_PX * 2 - 1,
            FIXATION_TOLERANCE_PX * 2 - 1,
            color=1
        )
        for x, y, width, height in self.button_rects:
            self.tracker.drawBox(
                x,
                y,
                width - 1,
                height - 1,
                color=1
            )
        for color_i, stim in enumerate(stims, 2):
            width = len(stim.text) * self.char_width
            height = self.char_height
            x, y = self.transform_to_top_left_origin(*stim.pos)
            self.tracker.drawBox(
                x - width // 2,
                y - height // 2,
                width - 1,
                height - 1,
                color=color_i
            )

    def show_fixation_dot(self):
        '''
        Show the fixation dot in the center of the screen.
        '''
        self.fixation_dot.draw()
        self.window.flip()

    def await_mouse_selection(self, buttons):
        '''
        Wait for a mouse click and then check if the click is on one of the
        object buttons; if so, return the selected item.
        '''
        self.mouse.clickReset()
        while True:
            core.wait(TIME_RESOLUTION_SECONDS)
            if self.mouse.getPressed()[0]:
                mouse_position = self.mouse.getPos()
                for button_i, button in enumerate(buttons):
                    if button.contains(mouse_position):
                        return button_i

    def await_gaze_selection(self, buttons):
        '''
        Wait for the participant to fixate an object for the specified time
        and return the selected item. If the C key is pressed, the trial will
        be abandoned in order to recalibrate.
        '''
        fixated_button_i = None
        gaze_timer = core.Clock()
        while True:
            if event.getKeys(['c']):
                raise InterruptTrialAndRecalibrate
            gaze_position = self.get_gaze_position()
            for button_i, button in enumerate(buttons):
                if button.contains(gaze_position):
                    if button_i == fixated_button_i:
                        # still looking at the same button
                        if gaze_timer.getTime() >= self.gaze_time:
                            # and gaze has been on button for sufficient time
                            return button_i
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
            keypresses = event.getKeys()
            if 'c' in keypresses:
                raise InterruptTrialAndRecalibrate
            elif 'q' in keypresses:
                raise InterruptTrialAndExit
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
                raise InterruptTrialAndRecalibrate
            if word_object.contains(self.get_gaze_position()):
                return True

    def await_boundary_cross(self):
        '''
        Wait for the participant's gaze to cross a boundary. If the C key
        is pressed, the trial will be abandoned in order to recalibrate.
        '''
        while True:
            if event.getKeys(['c']):
                raise InterruptTrialAndRecalibrate
            if self.get_gaze_position()[0] > self.boundary_x:
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

    def perform_predictor_test(self, test_item, mini_test=True):
        '''
        Show the predictors and wait for the participant to select one. Once a
        selection has been made, show the correct predictor within its
        phrase.
        '''
        pred_stim, targ_stim = self.mini_test_phrase_stims[test_item]
        if mini_test:
            self.missing_predictor_stim.draw()
            targ_stim.draw()
        for button, label in zip(self.predictor_buttons, self.predictor_button_labels):
            button.draw()
            label.draw()
        self.window.flip()
        # await predictor selection and show feedback
        if mini_test:
            selected_button_i = self.await_mouse_selection(self.predictor_buttons)
            pred_stim.draw()
            targ_stim.draw()
        else:
            selected_button_i = self.await_gaze_selection(self.predictor_buttons)
        selected_predictor = self.user_data['predictor_array'][selected_button_i]
        self.window.flip()
        core.wait(self.pause_time * 2)
        return selected_predictor

    def perform_object_test(self, test_item, mini_test=True):
        '''
        Show the objects and wait for the participant to select one. Once a
        selection has been made, show the correct object.
        '''
        pred_stim, targ_stim = self.mini_test_phrase_stims[test_item]
        if mini_test:
            if self.include_predictors:
                pred_stim.draw()
            targ_stim.draw()
        for object_button in self.object_buttons:
            object_button.draw()
        self.window.flip()
        # await object selection and show object feedback
        if mini_test:
            selected_button_i = self.await_mouse_selection(self.object_buttons)
            if self.include_predictors:
                pred_stim.draw()
            targ_stim.draw()
        else:
            selected_button_i = self.await_gaze_selection(self.object_buttons)
        selected_object = self.user_data['object_array'][selected_button_i]
        correct_button_i = self.user_data['object_array'].index(test_item)
        self.object_buttons[correct_button_i].draw()
        self.window.flip()
        core.wait(self.pause_time * 2)
        return selected_object

    def training_block(self, training_items, test_item):
        '''
        Run a block of training (passive exposure trials, followed by a
        mini-test).
        '''
        if SKIP_TRAINING:
            return
        # display each passive exposure trial
        for training_item in training_items:
            object_stim = self.training_object_stims[training_item]
            pred_stim, targ_stim = self.training_phrase_stims[training_item]
            # show the object stim
            object_stim.draw()
            self.window.flip()
            core.wait(self.pause_time)
            # show the predictor word
            if self.include_predictors:
                object_stim.draw()
                pred_stim.draw()
                self.window.flip()
                core.wait(self.pause_time)
            # show the target word
            object_stim.draw()
            if self.include_predictors:
                pred_stim.draw()
            targ_stim.draw()
            self.window.flip()
            core.wait(self.trial_time - self.pause_time)
            self.window.flip()
            core.wait(self.pause_time)
        # Perform mini test
        selected_predictor = None
        correct_predictor = None
        if self.predictor_tests:
            selected_predictor = self.perform_predictor_test(test_item, mini_test=True)
            correct_predictor = self.task_data['grammar'][test_item][0] == selected_predictor
            core.wait(self.pause_time)
        selected_object = self.perform_object_test(test_item, mini_test=True)
        correct_object = test_item == selected_object
        # save response
        self.store_response(f'mini_test', {
            'test_item': test_item,
            'selected_predictor': selected_predictor,
            'selected_object': selected_object,
            'correct_predictor': correct_predictor,
            'correct_object': correct_object,
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

    def free_fixation_test(self, test_item):
        '''
        Run a free-fixation trial. Participant must fixate a dot for 2
        seconds, after which a word is flashed up real quick in a random
        position on the screen. The participant is then shown the object
        stimuli and must select one by holding their gaze on it for 2
        seconds.
        '''
        if not TEST_MODE:
            self.mouse.setVisible(False)
        # determine position to place the word
        word_position_tl = self.generate_random_word_position()
        word_position = self.transform_to_center_origin(*word_position_tl)
        targ_stim = visual.TextStim(self.window,
            color='black',
            font='Courier New',
            pos=word_position,
            height=self.char_height,
            text=self.user_data['phrase_forms'][test_item][1],
        )
        # potentially perform calibration
        if self.n_trials_until_calibration == 0:
            self.perform_calibration()
        self.n_trials_until_calibration -= 1
        # update the experimenter's screen
        self.render_experimenter_screen([targ_stim])
        # initialize eye tracker recording
        if not TEST_MODE:
            self.tracker.startRecording(1, 1, 1, 1)
            self.tracker.drawText(
                f'"Trial {self.n_completed_trials + 1} ({self.n_trials_until_calibration})"'
            )
            self.tracker.sendMessage('trial_type free_fixation_test')
            self.tracker.sendMessage(f'test_item {test_item}')
        # show fixation dot and await fixation
        self.show_fixation_dot()
        self.await_fixation_on_fixation_dot()
        # show word stim
        targ_stim.draw()
        if not TEST_MODE:
            self.tracker.sendMessage('start_word_presentation')
        self.window.flip()
        # await entry into the word boundary
        self.await_word_entry(targ_stim)
        if not TEST_MODE:
            self.tracker.sendMessage('trigger_timer')
        # hide word after reveal time
        core.wait(self.reveal_time)
        self.window.flip()
        if not TEST_MODE:
            self.tracker.sendMessage('end_word_presentation')
        core.wait(self.pause_time)
        # show response buttons, await response, and show feedback
        selected_object = self.perform_object_test(test_item, mini_test=False)
        # stop recording and save response
        if not TEST_MODE:
            self.tracker.stopRecording()
        phrase_position = word_position_tl[0] - self.targ_word_width // 2, word_position_tl[1]
        self.store_response('free_fixation_test', {
            'test_item': test_item,
            'word_position': phrase_position,
            'selected_object': selected_object,
            'correct_object': selected_object == test_item,
        })
        self.n_completed_trials += 1

    def create_masked_stim(self, target_stim):
        target_form = target_stim.text
        mask_length = len(target_form) - 2
        random.shuffle(self.task_data['masking_alphabet'])
        mask = ''.join(self.task_data['masking_alphabet'][:mask_length])
        return visual.TextStim(self.window,
            color='black',
            font='Courier New',
            pos=target_stim.pos,
            height=target_stim.height,
            text=f'{target_form[0]}{mask}{target_form[-1]}',
        )

    def free_continuation_test(self, test_item, masked=True):
        '''
        Run a free-continuation trial. Participant must fixate a dot for 2
        seconds, after which the phrase appears with the predictor aligned
        with the participant's gaze position. Once the participant's gaze
        crosses the boundary a timer is started after which the phrase
        disappears. Finally, the participant is shown the object stimuli and
        must select one by holding their gaze on it for 2 seconds.
        '''
        if not TEST_MODE:
            self.mouse.setVisible(False)
        pred_stim, targ_stim = self.test_phrase_stims[test_item]
        if masked:
            masked_targ_stim = self.create_masked_stim(targ_stim)
        pred_position_tl = self.transform_to_top_left_origin(*pred_stim.pos)
        targ_position_tl = self.transform_to_top_left_origin(*targ_stim.pos)
        if self.n_trials_until_calibration == 0:
            self.perform_calibration()
        self.n_trials_until_calibration -= 1
        # update the experimenter's screen
        self.render_experimenter_screen([pred_stim, targ_stim])
        # initialize eye tracker recording
        if not TEST_MODE:
            self.tracker.startRecording(1, 1, 1, 1)
            self.tracker.drawText(
                f'"Trial {self.n_completed_trials + 1} ({self.n_trials_until_calibration})"'
            )
            self.tracker.sendMessage('trial_type free_continuation_test')
            self.tracker.sendMessage(f'test_item {test_item}')
        # show fixation dot and await fixation
        self.show_fixation_dot()
        self.await_fixation_on_fixation_dot()
        # show masked phrase stim
        pred_stim.draw()
        if masked:
            masked_targ_stim.draw()
        else:
            targ_stim.draw()
        if not TEST_MODE:
            self.tracker.sendMessage('start_word_presentation')
        self.window.flip()
        # draw the unmasked stim
        pred_stim.draw()
        targ_stim.draw()
        # await entry across the word boundary
        self.await_boundary_cross()
        if not TEST_MODE:
            self.tracker.sendMessage('trigger_timer')
        self.window.flip()
        # hide word after reveal time
        core.wait(self.reveal_time)
        self.window.flip()
        if not TEST_MODE:
            self.tracker.sendMessage('end_word_presentation')
        core.wait(self.pause_time)
        # show response buttons, await response, and show feedback
        selected_object = self.perform_object_test(test_item, mini_test=False)
        # stop recording and save response
        if not TEST_MODE:
            self.tracker.stopRecording()
        phrase_position = pred_position_tl[0] - self.pred_word_width // 2, pred_position_tl[1]
        self.store_response('free_continuation_test', {
            'test_item': test_item,
            'word_position': phrase_position,
            'selected_object': selected_object,
            'correct_object': selected_object == test_item,
        })
        self.n_completed_trials += 1

    def execute(self):
        '''
        Execute the experiment: Iterate over the trial sequence and run each
        trial. If the Q key is pressed during a trial, the experiment will be
        terminated at the end of the trial. If a trial completes
        successfully, the sequence position is incremented and the current
        user_data is saved. Once the experiment has been completed the eye
        tracker recording is saved.
        '''
        while self.user_data['sequence_position'] < len(self.user_data['trial_sequence']):
            if event.getKeys(['q']):
                break
            trial_type, params = self.user_data['trial_sequence'][self.user_data['sequence_position']]
            trial_func = getattr(self, trial_type)
            try:
                trial_func(**params)
            except InterruptTrialAndRecalibrate:
                self.abandon_trial()
                self.perform_calibration()
            except InterruptTrialAndExit:
                self.abandon_trial()
                break
            else:
                self.user_data['sequence_position'] += 1
                self.save_user_data()
        visual.TextStim(self.window,
            color='black',
            text=INSTRUCTION_END[LANGUAGE],
        ).draw()
        self.window.flip()
        self.save_tracker_recording(convert_to_asc=True)
        core.quit()


def generate_phrase_forms(task):
    '''
    Shuffle the alphabet and generate the words forms and phrases. Each
    participant sees a different set of surface forms, although the
    underlying structure is always the same.
    '''
    alphabet = []
    for symbols in task['alphabet']:
        random.shuffle(symbols)
        alphabet.extend(symbols)
    phrase_forms = []
    for predictor_i, word_template in task['grammar']:
        target_word_form = ''.join([alphabet[letter_index] for letter_index in word_template])
        if task['predictor_forms']:
            predictor_word_form = task['predictor_forms'][predictor_i]
        else:
            predictor_word_form = None
        phrase_forms.append((predictor_word_form, target_word_form))
    return phrase_forms

def generate_predictor_array(task):
    '''
    Generates a random order in which to present the predictors in the array.
    '''
    predictors = list(range(len(task['predictor_forms'])))
    random.shuffle(predictors)
    return predictors

def generate_object_array(task):
    '''
    Generates a random order in which to present the object stimuli in the
    array.
    '''
    objects = list(range(task['n_items']))
    random.shuffle(objects)
    return objects

def create_item_indices(frequency_distribution):
    '''
    Generate a list of indices according to some frequency distribution.
    '''
    item_indices = []
    for item_index, frequency in enumerate(frequency_distribution):
        item_indices.extend([item_index] * frequency)
    random.shuffle(item_indices)
    return item_indices

def generate_trial_sequence(task):
    '''
    Generates the main sequence of trials, including the training trials
    (passive exposure and mini-test) and both types of test.
    '''
    training_indices = create_item_indices(task['frequency_distribution'])
    test_indices = create_item_indices(task['frequency_distribution'])
    # TRAINING INSTRUCTIONS
    if task['predictor_tests']:
        trial_sequence = [('instructions', {
            'image': f'training_with_predictor_{LANGUAGE}.png',
        })]
    else:
        trial_sequence = [('instructions', {
            'image': f'training_{LANGUAGE}.png',
        })]
    # TRAINING TRIALS
    for i in range(task['training_reps']):
        for j in range(task['mini_test_freq']):
            training_items = []
            for k in range(task['n_items']):
                if not training_indices:
                    training_indices = create_item_indices(task['frequency_distribution'])
                training_items.append(training_indices.pop())
                if len(training_items) == task['mini_test_freq']:
                    if not test_indices:
                        test_indices = create_item_indices(task['frequency_distribution'])
                    test_item = test_indices.pop()
                    trial_sequence.append(('training_block', {
                        'training_items': training_items,
                        'test_item': test_item,
                    }))
                    training_items = []
    
    # FREE-FIXATION TEST TRIALS
    if task['free_fixation_reps'] > 0:
        trial_sequence.append(('instructions', {
            'image': f'free_fixation_test_{LANGUAGE}.png',
        }))
        test_indices = create_item_indices(task['frequency_distribution'])
        test_trials = []
        for i in range(task['free_fixation_reps']):
            for j in range(task['n_items']):
                if not test_indices:
                    test_indices = create_item_indices(task['frequency_distribution'])
                test_item = test_indices.pop()
                test_trials.append(('free_fixation_test', {
                    'test_item': test_item,
                }))
        trial_sequence.extend(test_trials)

    # FREE-CONTINNUATIOIN TEST TRIALS
    if task['free_continuation_reps'] > 0:
        trial_sequence.append(('instructions', {
            'image': f'free_continuation_test_{LANGUAGE}.png',
        }))
        test_indices = create_item_indices(task['frequency_distribution'])
        test_trials = []
        for i in range(task['free_continuation_reps']):
            for j in range(task['n_items']):
                if not test_indices:
                    test_indices = create_item_indices(task['frequency_distribution'])
                test_item = test_indices.pop()
                test_trials.append(('free_continuation_test', {
                    'test_item': test_item,
                    'masked': bool(task['masking_alphabet']),
                }))
        trial_sequence.extend(test_trials)

    return trial_sequence


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', action='store', type=str, help='task ID')
    parser.add_argument('user_id', action='store', type=str, help='user ID')
    args = parser.parse_args()

    Experiment(args.task_id, args.user_id).execute()
