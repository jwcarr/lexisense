// Extract prolific ID from the URL
const user_id = (new URL(window.location.href)).searchParams.get('PROLIFIC_PID');

// Credit card sizes and their corresponding letter sizes. For example, if the
// credit card is set to a size of 213x135, then letters will be displayed at
// a size of 25x50.
const card_sizes = [[213.0, 135.0], [221.52, 140.4], [230.04, 145.8], [238.56, 151.2], [247.08, 156.6], [255.6, 162.0], [264.12, 167.4], [272.64, 172.8], [281.16, 178.2], [289.68, 183.6], [298.2, 189.0], [306.72, 194.4], [315.24, 199.8], [323.76, 205.2], [332.28, 210.6], [340.8, 216.0], [349.32, 221.4], [357.84, 226.8], [366.36, 232.2], [374.88, 237.6], [383.4, 243.0], [391.92, 248.4], [400.44, 253.8], [408.96, 259.2], [417.48, 264.6], [426.0, 270.0], [434.52, 275.4], [443.04, 280.8], [451.56, 286.2], [460.08, 291.6], [468.6, 297.0], [477.12, 302.4], [485.64, 307.8], [494.16, 313.2], [502.68, 318.6], [511.2, 324.0], [519.72, 329.4], [528.24, 334.8], [536.76, 340.2], [545.28, 345.6], [553.8, 351.0], [562.32, 356.4], [570.84, 361.8], [579.36, 367.2], [587.88, 372.6], [596.4, 378.0], [604.92, 383.4], [613.44, 388.8], [621.96, 394.2], [630.48, 399.6], [639.0, 405.0]];
const letter_sizes = [[25, 50], [26, 52], [27, 54], [28, 56], [29, 58], [30, 60], [31, 62], [32, 64], [33, 66], [34, 68], [35, 70], [36, 72], [37, 74], [38, 76], [39, 78], [40, 80], [41, 82], [42, 84], [43, 86], [44, 88], [45, 90], [46, 92], [47, 94], [48, 96], [49, 98], [50, 100], [51, 102], [52, 104], [53, 106], [54, 108], [55, 110], [56, 112], [57, 114], [58, 116], [59, 118], [60, 120], [61, 122], [62, 124], [63, 126], [64, 128], [65, 130], [66, 132], [67, 134], [68, 136], [69, 138], [70, 140], [71, 142], [72, 144], [73, 146], [74, 148], [75, 150]];
const n_letter_slots = 13;
const center_index = (n_letter_slots - 1) / 2;

// Participant's randomized alphabet and object array will be stored as globals
let font, alphabet, object_array;


function iterAtInterval(iterable, interval, func, final_func) {
	// Call func on each item in an iterable with a given time interval. Once all
	// items have been iterated over, call final_func.
	if (iterable.length === 0) {
		final_func();
	} else {
		setTimeout(function() {
			iterAtInterval(iterable.slice(1), interval, func, final_func);
		}, interval);
		func(iterable[0]);
	}
}

function updateProgress(progress) {
	$('#progress').animate({width: progress*900}, 250);
}

function updateBonus(total_bonus) {
	$('#bonus_amount').html(total_bonus);
}

function preloadObject(object_id) {
	hideObject();
	$('#object_image').attr('src', 'images/objects/' + object_id + '.png');
}

function showObject() {
	$('#object').show();
}

function hideObject() {
	$('#object').hide();
}

function preloadWord(word, fixation_position=null) {
	hideWord();
	if (fixation_position === null)
		fixation_position = Math.floor(word.length / 2);
	const start = center_index - fixation_position;
	const end = start + word.length;
	for (let i = 0; i < n_letter_slots; i++) {
		if (i >= start && i < end)
			$('#char'+i).attr('src', 'images/' + font + '/' + alphabet[word[i-start]] + '.png');
		else
			$('#char'+i).attr('src', 'images/empty_char.png');
	}
}

function showWord() {
	$('#word').show();
}

function hideWord() {
	$('#word').hide();
	for (let i = 0; i < n_letter_slots; i++) {
		$('#char'+i).attr('src', 'images/empty_char.png');
	}
}

function showObjectButtons() {
	$('#button_panel').show();
}

function hideObjectButtons() {
	$('#button_panel').hide();
	for (let object_i = 0; object_i < 8; object_i++) {
		$('#object'+object_i).show();
	}
}

function showFeedback(target_object) {
	const object_button = object_array.indexOf(target_object);
	for (let object_i = 0; object_i < 8; object_i++) {
		if (object_i != object_button)
			$('#object'+object_i).hide();
	}
}

function showFixationCross() {
	$('#fixation_cross').show();
}

function hideFixationCross() {
	$('#fixation_cross').hide();
}

function enableButton(button_id) {
	$(button_id).css('background-color', 'black');
	$(button_id).css('cursor', 'pointer');
	$(button_id).attr('disabled', false);
}

function disableButton(button_id) {
	$(button_id).css('background-color', 'gray');
	$(button_id).css('cursor', 'default');
	$(button_id).attr('disabled', true);
}

function setDisplaySize(size_selection) {
	const letter_width = letter_sizes[size_selection][0];
	const letter_height = letter_sizes[size_selection][1];
	const left_edge = 500 - (letter_width / 2) - (center_index * letter_width);
	for (let i = 0; i < n_letter_slots; i++) {
		$('#char'+i).css('width', letter_width + 'px');
		$('#char'+i).css('height', letter_height + 'px');
		$('#char'+i).css('left', left_edge + i * letter_width + 'px');
	}
	$('#cross').css('width', letter_width + 'px');
	$('#cross').css('height', letter_height + 'px');
	$('#cross').css('left', 500 - (letter_width/2) + 'px');
	$('#object_image').css('width', letter_height * 2 + 'px');
	$('#object_image').css('height', letter_height * 2 + 'px');
	$('#object_image').css('left', 500 - letter_height + 'px');
}


// Establish websocket connection with the server using socket.io
const socket = io.connect();

// Server 
socket.on('initialize', function(payload) {
	font = payload.font;
	alphabet = payload.alphabet;
	object_array = payload.object_array;
	let alphabet_preload_html;
	for (let i = 0; i < payload.alphabet.length; i++) {
		alphabet_preload_html += '<img src="images/' + font + '/' + alphabet[i] + '.png" />';
	}
	$('#alphabet_preloader').html(alphabet_preload_html);
	for (let i = 0; i < object_array.length; i++) {
		$('#object_image' + i).attr('src', 'images/objects/' + object_array[i] + '.png');
	}
	updateBonus(payload.total_bonus);
	setDisplaySize(payload.size_selection);
	$('#header').show();
	socket.emit('next', {user_id, 'initialization':true});
});

socket.on('consent', function(payload) {
	$('#consent_session_time').html(payload.session_time);
	$('#consent_basic_pay').html('£' + (payload.basic_pay/100).toFixed(2));
	$('#consent_max_pay').html('£' + (payload.max_pay/100).toFixed(2));
	$('#submit_consent').click(function() {
		$('#submit_consent').off('click');
		$('#consent').hide();
		$('#header').show();
		socket.emit('next', {user_id});
	});
	$('#confirm_consent').click(function() {
		if ($('#confirm_consent').is(':checked'))
			enableButton('#submit_consent');
		else
			disableButton('#submit_consent');
	});
	disableButton('#submit_consent');
	$('#consent').show();
});

socket.on('calibration', function(payload) {
	let size_selection = 15;
	updateProgress(payload.progress);
	$('#increase').click(function() {
		if (size_selection === 50)
			return;
		size_selection += 1;
		$('#card').attr('width', card_sizes[size_selection][0]);
		$('#card').attr('height', card_sizes[size_selection][1]);
	});
	$('#decrease').click(function() {
		if (size_selection === 0)
			return;
		size_selection -= 1;
		$('#card').attr('width', card_sizes[size_selection][0]);
		$('#card').attr('height', card_sizes[size_selection][1]);
	});
	$('#confirm_calibration').click(function() {
		$('#set_size').off('click');
		$('#calibration').hide();
		setDisplaySize(size_selection);
		socket.emit('next', {user_id, size_selection});
	});
	$('#calibration').show();
});

socket.on('distance_instructions', function(payload) {
	updateProgress(payload.progress);
	$('#confirm_distance').click(function() {
		$('#confirm_distance').off('click');
		$('#distance_instructions').hide();
		socket.emit('next', {user_id});
	});
	$('#distance_instructions').show();
});

socket.on('training_instructions', function(payload) {
	updateProgress(payload.progress);
	$('#start_training').click(function() {
		$('#start_training').off('click');
		$('#training_instructions').hide();
		socket.emit('next', {user_id});
	});
	setTimeout(function() {
		enableButton('#start_training');
	}, payload.instruction_time);
	disableButton('#start_training');
	$('#training_instructions').show();
});

socket.on('test_instructions', function(payload) {
	updateProgress(payload.progress);
	$('#start_test').click(function() {
		$('#start_test').off('click');
		$('#test_instructions').hide();
		$('#experiment').show();
		socket.emit('next', {user_id});
	});
	setTimeout(function() {
		enableButton('#start_test');
	}, payload.instruction_time);
	disableButton('#start_test');
	$('#test_instructions').show();
});

socket.on('training_block', function(payload) {
	updateProgress(payload.progress);
	iterAtInterval(payload.training_trials, payload.trial_time,
		// func: On each passive exposure trial...
		function(trial) {
			$('#word').css('top', 350);
			// 1. Preload the object and word
			preloadObject(trial.object);
			preloadWord(trial.word);
			setTimeout(function() {
				setTimeout(function() {
					// 3. After pause_time, show the word
					showWord();
				}, payload.pause_time);
				// 2. After pause_time, show the object
				showObject();
			}, payload.pause_time);
		},
		// final_func: On each mini-test trial...
		function() {
			$('#word').css('top', 200);
			// 1. Preload the test word
			hideObject();
			preloadWord(payload.test_trial.word);
			setTimeout(function() {
				$('button[id^="object"]').click(function() {
					$('button[id^="object"]').off('click');
					const reaction_time = Math.floor(performance.now() - start_time);
					const clicked_button = parseInt($(this).attr('id').match(/object(.)/)[1]);
					const selected_object = object_array[clicked_button];
					setTimeout(function() {
						// 4. After 2*pause_time, hide the word and object buttons and request the next trial
						hideWord();
						hideObjectButtons();
						socket.emit('next', {user_id, response:{
							test_type : 'mini_test',
							object : payload.test_trial.object,
							selected_object,
							reaction_time,
						}});
					}, payload.pause_time * 2);
					// 3. On button click, show feedback and update the user's bonus
					showFeedback(payload.test_trial.object);
					if (selected_object === payload.test_trial.object)
						updateBonus(payload.total_bonus + 1);
				});
				// 2. After pause_time, show the test word and the object buttons
				showWord();
				showObjectButtons();
				const start_time = performance.now();
			}, payload.pause_time);
		}
	);
	$('#experiment').show();
});

socket.on('ovp_test', function(payload) {
	updateProgress(payload.progress);
	$('#word').css('top', 200);
	// 1. Show fixation cross and preload the test word
	showFixationCross();
	preloadWord(payload.word, payload.fixation_position);
	setTimeout(function() {
		setTimeout(function() {
			$('button[id^="object"]').click(function() {
				$('button[id^="object"]').off('click');
				const reaction_time = Math.floor(performance.now() - start_time);
				const clicked_button = parseInt($(this).attr('id').match(/object(.)/)[1]);
				const selected_object = object_array[clicked_button];
				setTimeout(function() {
					// 5. After 2*pause_time, hide the object buttons and request the next trial
					hideObjectButtons();
					socket.emit('next', {user_id, response:{
						test_type : 'ovp_test',
						fixation_position : payload.fixation_position,
						object : payload.object,
						selected_object,
						reaction_time,
					}});
				}, payload.pause_time * 2);
				// 4. On button click, show feedback and update the user's bonus
				showFeedback(payload.object);
				if (selected_object === payload.object)
					updateBonus(payload.total_bonus + 1);
			});
			// 3. After reveal_time, hide the word and show the object buttons
			hideWord();
			showObjectButtons();
			const start_time = performance.now();
		}, payload.reveal_time);
		// 2. After delay_time, reveal the word
		hideFixationCross();
		showWord();
	}, payload.delay_time);
	$('#experiment').show();
});

socket.on('questionnaire', function(payload) {
	updateProgress(payload.progress);
	$('#experiment').hide();
	$('#submit_questionnaire').click(function() {
		$('#submit_questionnaire').off('click');
		const comments = $('#comments').val();
		socket.emit('next', {user_id, comments});
	});
	$('#comments').keyup(function() {
		if ($(this).val().length > 0)
			enableButton('#submit_questionnaire');
		else
			disableButton('#submit_questionnaire');
	});
	disableButton('#submit_questionnaire');
	$('#questionnaire').show();
	$('#comments').focus();
});

socket.on('end_of_experiment', function(payload) {
	updateProgress(payload.progress);
	$('#questionnaire').hide();
	$('#basic_pay').html('£' + (payload.basic_pay/100).toFixed(2));
	$('#bonus_pay').html('£' + (payload.total_bonus/100).toFixed(2));
	$('#total_pay').html('£' + ((payload.basic_pay + payload.total_bonus)/100).toFixed(2));
	$('#exit').click(function() {
		$('#exit').off('click');
		$('#exit').hide();
		window.location.href = payload.return_url;
	});
	$('#end_of_experiment').show();
});

socket.on('report', function(payload) {
	$('#status_message').html(payload.message);
	$('#task_status').show();
});

$(document).ready(function() {
	if (!/Android|webOS|iPhone|iPad|iPod|IEMobile|Opera Mini/i.test(navigator.userAgent))
		socket.emit('handshake', {user_id});
});
