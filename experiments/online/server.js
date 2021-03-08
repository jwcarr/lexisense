// ------------------------------------------------------------------
// Parameters
// ------------------------------------------------------------------

// Name used for the MongoDB database
const DATABASE_NAME = 'ovp';

// Regex defining valid Prolific participant IDs (24 hex digits)
const VALID_USER_ID = /^[a-f0-9]{24}$/;

// Use http or https as the protocol
const PROTOCOL = 'https';

// Port number to listen on
const PORT = 8080;

// If https, provide paths to the SSL encryption keys
const SSL_KEY_FILE = '/etc/letsencrypt/live/joncarr.net/privkey.pem';
const SSL_CERT_FILE = '/etc/letsencrypt/live/joncarr.net/fullchain.pem';

// ------------------------------------------------------------------
// Import modules
// ------------------------------------------------------------------

const fs = require('fs');
const http = require(PROTOCOL);
const express = require('express');
const mongodb = require('mongojs');
const socketio = require('socket.io');

// ------------------------------------------------------------------
// Server setup
// ------------------------------------------------------------------

const app = express();
app.use(express.static(`${__dirname}/client`));

const config = {};
if (PROTOCOL === 'https') {
	config.key = fs.readFileSync(SSL_KEY_FILE);
	config.cert = fs.readFileSync(SSL_CERT_FILE);
}
const server = http.createServer(config, app);

const db = mongodb(DATABASE_NAME, ['tasks', 'users']);

const socket = socketio(server);

// ------------------------------------------------------------------
// Functions
// ------------------------------------------------------------------

function range(start_inclusive, end_exclusive) {
	if (end_exclusive === undefined) {
		end_exclusive = start_inclusive;
		start_inclusive = 0;
	}
	const array = [];
	for (let i = start_inclusive; i < end_exclusive; i++) {
		array.push(i);
	}
	return array;
}

function randInt(start_inclusive, end_exclusive) {
	if (end_exclusive === undefined) {
		end_exclusive = start_inclusive;
		start_inclusive = 0;
	}
	return Math.floor(Math.random() * (end_exclusive - start_inclusive) + start_inclusive);
}

function shuffle(array) {
	let counter = array.length, temp, index;
	while (counter) {
		index = randInt(counter--);
		temp = array[counter];
		array[counter] = array[index];
		array[index] = temp;
	}
}

function getCurrentTime() {
	return Math.floor(new Date() / 1000);
}

function generateAlphabet(task) {
	const alphabet = [...task.alphabet];
	shuffle(alphabet);
	return alphabet;
}

function generateObjectArray(task) {
	const objects = range(task.n_items);
	shuffle(objects);
	return objects;
}

function generateTrialSequence(task) {
	const item_indices = range(task.n_items), seen_items = [], trial_sequence = [];
	trial_sequence.push({event:'consent', payload:{
		session_time : task.session_time,
		basic_pay : task.basic_pay,
		max_pay : task.max_pay,
		progress : 5,
	}});
	trial_sequence.push({event:'calibration', payload:{
		progress : 10,
	}});
	trial_sequence.push({event:'distance_instructions', payload:{
		progress : 5,
	}});
	trial_sequence.push({event:'training_instructions', payload:{
		instruction_time : task.instruction_time,
		progress : 10,
	}});
	for (let i = 0; i < task.training_reps; i++) {
		for (let j = 0; j < task.mini_test_freq; j++) {
			let training_trials = [];
			shuffle(item_indices);
			for (let k = 0; k < task.n_items; k++) {
				const training_item = item_indices[k];
				const training_trial = {word:task.words[training_item], object:training_item};
				if (j === 0)
					seen_items.push(training_item);
				training_trials.push(training_trial);
				if (training_trials.length === task.mini_test_freq) {
					const test_item = seen_items.splice(randInt(seen_items.length),1)[0];
					const test_trial = {word:task.words[test_item], object:test_item};
					trial_sequence.push({event:'training_block', payload:{
						training_trials,
						test_trial,
						trial_time : task.trial_time,
						pause_time : task.pause_time,
						progress : task.mini_test_freq + 1,
					}});
					training_trials = [];
				}
			}
		}
	}
	trial_sequence.push({event:'test_instructions', payload:{
		instruction_time : task.instruction_time,
		progress : 10,
	}});
	for (let i = 0; i < task.test_reps; i++) {
		let test_trials = [];
		for (let j = 0; j < task.n_items; j++) {
			for (let fixation_position = 0; fixation_position < task.n_letters; fixation_position++) {
				test_trials.push({event:'ovp_test', payload:{
					word : task.words[j],
					object : j,
					fixation_position,
					reveal_time : task.reveal_time,
					delay_time : randInt(task.delay_min_time, task.delay_max_time),
					pause_time : task.pause_time,
					progress : 2,
				}});
			}
		}
		shuffle(test_trials);
		for (let trial of test_trials) {
			trial_sequence.push(trial);
		}
		test_trials = [];
	}
	trial_sequence.push({event:'questionnaire', payload:{
		progress : 10,
	}});
	trial_sequence.push({event:'end_of_experiment', payload:{
		return_url : task.return_url,
		basic_pay : task.basic_pay,
		progress : 0,
	}});
	let total_units_of_progress = 0;
	for (let i = 0; i < trial_sequence.length; i++) {
		total_units_of_progress += trial_sequence[i].payload.progress;
		trial_sequence[i].payload.progress = total_units_of_progress - trial_sequence[i].payload.progress;
	}
	for (let i = 0; i < trial_sequence.length; i++) {
		trial_sequence[i].payload.progress = trial_sequence[i].payload.progress / total_units_of_progress;
	}
	return trial_sequence;
}

function reportError(client, error_number, reason) {
	const message = 'Error ' + error_number + ': ' + reason;
	console.log(getCurrentTime() + ' ' + message);
	return client.emit('report', {message});
}

// ------------------------------------------------------------------
// Client connection handlers
// ------------------------------------------------------------------

socket.on('connection', function(client) {

	// Client makes initial handshake
	client.on('handshake', function(payload) {
		// Check for a valid user ID
		if (!VALID_USER_ID.test(payload.user_id))
			return reportError(client, 117, 'Unable to validate participant ID.');
		// Attempt to find the user in the database
		db.users.findOne({user_id: payload.user_id}, function(err, user) {
			if (err)
				return reportError(client, 118, 'Unable to validate participant ID.');
			// If we've seen this user before...
			if (user) {
				if (user.status != 'active')
					return reportError(client, 116, 'You have already completed this task.');
				// Reinitialize the user and make a note of this in the database
				db.users.update({user_id: user.user_id}, {$inc: {n_reinitializations: 1}});
				return client.emit('initialize', {
					alphabet: user.alphabet,
					object_array: user.object_array,
					size_selection: user.size_selection,
					total_bonus: user.total_bonus,
				});
			}
			// If we haven't seen this user before...
			// Check to see which tasks are active and sort them by the number of
			// participants that have taken part so far.
			db.tasks.find({status: 'active'}).sort({n_participants: 1}, function(err, tasks) {
				if (err || tasks.length === 0)
					return reportError(client, 119, 'No task available.');
				// Pick the first task (i.e. with the fewest participants)
				const task = tasks[0];
				// Create user with a random alphabet, object array, and trial sequence
				const time = getCurrentTime();
				const user = {
					user_id: payload.user_id,
					task_id: task.task_id,
					creation_time: time,
					modified_time: time,
					status: 'active',
					alphabet: generateAlphabet(task),
					object_array: generateObjectArray(task),
					trial_sequence: generateTrialSequence(task),
					sequence_position: 0,
					responses: [],
					size_selection: 25,
					basic_pay: task.basic_pay,
					total_bonus: 0,
					comments: null,
				};
				// Save the new user to the database
				db.users.save(user, function(err, saved) {
					if (err || !saved)
						return reportError(client, 121, 'This task is currently unavailable.');
					// Increment the number of participants on this task
					db.tasks.update({_id: task._id}, {$inc: {n_participants: 1}});
					// Tell the client to initialize
					return client.emit('initialize', {
						alphabet: user.alphabet,
						object_array: user.object_array,
						size_selection: user.size_selection,
						total_bonus: user.total_bonus,
					});
				});
			});
		});
	});

	// Client requests next trial
	client.on('next', function(payload) {
		// First determine what needs to be updated in the database
		const time = getCurrentTime();
		const update = {
			$set : {modified_time: time},
			$inc : {sequence_position: 1},
		};
		if (payload.initialization)
			update.$inc.sequence_position = 0;
		if (payload.size_selection != undefined)
			update.$set.size_selection = payload.size_selection;
		if (payload.comments)
			update.$set.comments = payload.comments;
		if (payload.response) {
			payload.response.time = time;
			update.$push = {responses:payload.response};
			if (payload.response.object === payload.response.selected_object)
				update.$inc.total_bonus = 1;
		}
		// Then update the database and send the next trial back down to the client
		db.users.findAndModify({query: {user_id: payload.user_id}, update, new: true}, function(err, user, last_err) {
			if (err || !user)
				return reportError(client, 126, 'Unrecognized participant ID.');
			if (user.status != 'active')
				return reportError(client, 127, 'Your session is no longer active.');
			const next = user.trial_sequence[user.sequence_position];
			next.payload.total_bonus = user.total_bonus;
			if (next.event === 'end_of_experiment')
				db.users.update({user_id: user.user_id}, {$set: {status: 'completed'}});
			return client.emit(next.event, next.payload);
		});
	});

});

server.listen(PORT);
