'''

This code is used to monitor the online experiments by interacting with the
MongoDB database. It can be used from the command line, e.g.:

	python mission_control.py some_task_id --status

of from a Python console:

	import mission_control as mc
	mc.status('some_task_id')

If the database is on a remote server, you can either run this module on the
remote or create an SSH tunnel, so that, e.g., port 27017 on your local
machine is mapped to port 27017 on the remote, e.g.:

	ssh -L 27017:localhost:27017 jon@joncarr.net

The general workflow is:

- launch: launch a task (as defined by the parameters in a JSON file)
- status: monitor the status of a task
- start/stop: make a task active/inactive
- exclude: exclude a dropout participant and increment the slots on the task
- approve/bonus: print an approval or bonusing list to submit to Prolific
- pull: pull down the data from all completed participants on a task

'''

from pathlib import Path
import json
import csv
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.json_util import dumps


# Data directory
ROOT_DIR = Path(__file__).absolute().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'experiments' / 'online'

# MongoDB connection
DOMAIN = 'localhost'
PORT = 27017
DATABASE_NAME = 'ovp'


db = MongoClient(DOMAIN, PORT)[DATABASE_NAME]


def status(task_id):
	n_slots = db.tasks.find_one({'task_id': task_id})['n_participants']
	status_count = {'active': 0, 'completed': 0, 'excluded': 0}
	for user in db.users.find({'task_id': task_id}):
		status_count[user['status']] += 1
		if user['status'] == 'active':
			percent_complete = int((user['sequence_position'] + 1) / len(user['trial_sequence']) * 100)
			sitting_time = timedelta(seconds=int(datetime.now().timestamp()) - user['modified_time'])
			print('-', user['user_id'], user['status'], f'{percent_complete}% ({sitting_time})')
		else:
			print('-', user['user_id'], user['status'])

	print('ACTIVE:', status_count['active'])
	print('COMPLETED:', status_count['completed'])
	print('EXCLUDED:', status_count['excluded'])
	print('OPEN SLOTS:', n_slots)

def launch(task_id, return_url):
	task_file = DATA_DIR / f'{task_id}.json'
	with open(task_file) as file:
		task = json.load(file)
	assert task_id == task['task_id']
	assert len(task['words']) == task['n_items']
	assert task['n_items'] % task['mini_test_freq'] == 0
	assert db.tasks.count_documents({'task_id':task['task_id']}) == 0
	task['max_pay'] = task['basic_pay'] + (task['training_reps'] * task['n_items'] + task['test_reps'] * task['n_letters'] * task['n_items'])
	task['status'] = 'active'
	task['return_url'] = return_url
	db.tasks.insert_one(task)
	print('Launched task:', task['task_id'])

def remove(task_id):
	if db.tasks.count_documents({'task_id':task_id}) == 0:
		raise ValueError('There is no task with this ID.')
	db.tasks.delete_one({'task_id': task_id})
	print('Removed task:', task_id)

def start(task_id):
	if db.tasks.count_documents({'task_id':task_id}) == 0:
		raise ValueError('There is no task with this ID.')
	db.tasks.update_one({'task_id': task_id}, {'$set':{'status':'active'}})
	print('Started task:', task_id)

def stop(task_id):
	if db.tasks.count_documents({'task_id':task_id}) == 0:
		raise ValueError('There is no task with this ID.')
	db.tasks.update_one({'task_id': task_id}, {'$set':{'status':'inactive'}})
	print('Stopped task:', task_id)

def approve(task_id):
	print('---------- START ----------')
	for user in db.users.find({'task_id': task_id}):
		if user['status'] == 'completed':
			print(user['user_id'])
	print('----------- END -----------')

def bonus(task_id):
	print('---------- START ----------')
	for user in db.users.find({'task_id': task_id}):
		if user['status'] == 'completed':
			pounds = str(user["total_bonus"] // 100)
			pence = str(user["total_bonus"] % 100).zfill(2)
			print(f'{user["user_id"]},{pounds}.{pence}')
	print('----------- END -----------')

def pull(task_id):
	data_dir = DATA_DIR / task_id
	if not data_dir.exists():
		data_dir.mkdir()
	demographic_data = get_demographic_data(task_id)
	annonymous_user_id = 1
	for user in db.users.find({'task_id': task_id}):
		if user['status'] == 'completed':
			if demographic_data:
				first_language = demographic_data[user['user_id']]['First Language']
				fluent_languages = set(demographic_data[user['user_id']]['Fluent languages'].split(', '))
				try:
					fluent_languages.remove(first_language)
				except KeyError:
					pass
				user['first_language'] = first_language if first_language != 'DATA EXPIRED' else None
				user['other_languages'] = sorted(list(fluent_languages))
			del user['_id']
			user['user_id'] = str(annonymous_user_id).zfill(2)
			with open(data_dir / f'{user["user_id"]}.json', 'w') as file:
				file.write(dumps(user, indent='\t'))
			annonymous_user_id += 1

def exclude(user_id):
	user = db.users.find_one_and_update({'user_id': user_id}, {'$set':{'status':'excluded'}})
	if user['status'] in ['active', 'completed']:
		db.tasks.update_one({'task_id': user['task_id']}, {'$inc':{'n_participants':1}})

def skip(user_id):
	db.users.update_one({'user_id': user_id}, {'$set':{'sequence_position':67, 'status':'active'}})

def get_demographic_data(task_id):
	exp_id = task_id.split('_')[0]
	csv_path = ROOT_DIR / 'private' / 'prolific_demographics' / f'{exp_id}.csv'
	if not csv_path.exists():
		return None
	demographic_data = {}
	with open(csv_path) as file:
		for user_data in csv.DictReader(file):
			demographic_data[user_data['participant_id']] = user_data
	return demographic_data


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('id', action='store', help='Task ID or user ID as appropriate')
	parser.add_argument('--launch', action='store', help='Add the task to the database (provide the return URL)')
	parser.add_argument('--status', action='store_true', help='Get a status update on the task')
	parser.add_argument('--remove', action='store_true', help='Remove the task from the database')
	parser.add_argument('--start', action='store_true', help='Set the task\'s status to active')
	parser.add_argument('--stop', action='store_true', help='Set the task\'s status to inactive')
	parser.add_argument('--approve', action='store_true', help='Print an approval list to submit to Prolific')
	parser.add_argument('--bonus', action='store_true', help='Print a bulk bonus list to submit to Prolific')
	parser.add_argument('--pull', action='store_true', help='Download data of all completed users on the task')
	parser.add_argument('--exclude', action='store_true', help='Exclude a user and open a new slot on their assigned task')
	parser.add_argument('--skip', action='store_true', help='Skip a user forward to the test phase (for testing purposes)')
	args = parser.parse_args()

	if args.status:
		status(args.id)
	elif args.launch:
		launch(args.id, args.launch)
	elif args.remove:
		remove(args.id)
	elif args.start:
		start(args.id)
	elif args.stop:
		stop(args.id)
	elif args.approve:
		approve(args.id)
	elif args.bonus:
		bonus(args.id)
	elif args.pull:
		pull(args.id)
	elif args.exclude:
		exclude(args.id)
	elif args.skip:
		skip(args.id)
	else:
		for task in db.tasks.find({'status': args.id}):
			print(task['task_id'])
