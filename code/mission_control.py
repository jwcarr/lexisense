'''

If the database is on a remote server, rather than connect to the database
directly, you can create an SSH tunnel, so that, e.g., port 27017 on your
local machine is mapped to port 27017 on the remote:

	ssh -L 27017:localhost:27017 jon@joncarr.net

'''

import pathlib
import json
from datetime import datetime
from pymongo import MongoClient
from bson.json_util import dumps


DATA_DIRECTORY = 'data/experiments/online/'
DOMAIN = 'localhost'
PORT = 27017
DATABASE_NAME = 'ovp'


def status(db):
	for task in db.tasks.find({}):
		if task['status'] == 'active':
			print(task['task_id'], task['n_participants'])
			for user in db.users.find({'task_id': task['task_id']}):
				percent_complete = int((user['sequence_position'] + 1) / len(user['trial_sequence']) * 100)
				modified_time = datetime.utcfromtimestamp(user['modified_time']).strftime('%Y-%m-%d %H:%M:%S')
				print('-', user['user_id'], modified_time, user['status'], f'{percent_complete}%')

def launch_task(db, task_id):
	task_file = pathlib.Path(DATA_DIRECTORY) / f'{task_id}.json'
	with open(task_file) as file:
		task = json.load(file)
	assert task_id == task['task_id']
	assert len(task['words']) == task['n_items']
	assert task['n_items'] % task['mini_test_freq'] == 0
	assert db.tasks.count_documents({'task_id':task['task_id']}) == 0
	task['max_pay'] = task['basic_pay'] + (task['training_reps'] * task['n_items'] + task['test_reps'] * task['n_letters'] * task['n_items'])
	db.tasks.insert_one(task)
	print('Launched task:', task['task_id'])

def remove_task(db, task_id):
	if db.tasks.count_documents({'task_id':task_id}) == 0:
		raise ValueError('There is no task with this ID.')
	db.tasks.delete_one({'task_id': task_id})
	print('Removed task:', task_id)

def start_task(db, task_id):
	if db.tasks.count_documents({'task_id':task_id}) == 0:
		raise ValueError('There is no task with this ID.')
	db.tasks.update_one({'task_id': task_id}, {'$set':{'status':'active'}})
	print('Started task:', task_id)

def stop_task(db, task_id):
	if db.tasks.count_documents({'task_id':task_id}) == 0:
		raise ValueError('There is no task with this ID.')
	db.tasks.update_one({'task_id': task_id}, {'$set':{'status':'inactive'}})
	print('Stopped task:', task_id)

def exclude_user(db, user_id):
	user = db.users.find_one_and_update({'user_id': user_id}, {'$set':{'status':'excluded'}})
	if user['status'] in ['active', 'completed']:
		db.tasks.update_one({'task_id': user['task_id']}, {'$inc':{'n_participants':-1}})

def pull_task(db, task_id):
	data_dir = pathlib.Path(DATA_DIRECTORY) / task_id
	if not data_dir.exists():
		data_dir.mkdir()
	annonymous_user_id = 0
	for user in db.users.find({'task_id': task_id}):
		if user['status'] == 'completed':
			del user['_id']
			user['user_id'] = str(annonymous_user_id)
			with open(data_dir / f'{user["user_id"]}.json', 'w') as file:
				file.write(dumps(user, indent='\t'))
			annonymous_user_id += 1

def erase_all_tasks(db):
	db.tasks.delete_many({})

def erase_all_users(db):
	db.users.delete_many({})


if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--status', action='store_true', help='Get a status update on all active tasks')
	parser.add_argument('--launch', action='store', type=str, help='ID of a task to launch')
	parser.add_argument('--remove', action='store', type=str, help='ID of a task to remove from the database')
	parser.add_argument('--start', action='store', type=str, help='ID of a task whose status should be set to active')
	parser.add_argument('--stop', action='store', type=str, help='ID of a task whose status should be set to inactive')
	parser.add_argument('--pull', action='store', type=str, help='Pull all users who have a completed a given task ID')
	parser.add_argument('--exclude', action='store', type=str, help='ID of a user to exclude')
	args = parser.parse_args()
	
	db = MongoClient(DOMAIN, PORT)[DATABASE_NAME]

	if args.status:
		status(db)
		exit()

	if args.launch:
		launch_task(db, args.launch)
		exit()

	if args.remove:
		remove_task(db, args.remove)
		exit()

	if args.start:
		start_task(db, args.start)
		exit()

	if args.stop:
		stop_task(db, args.stop)
		exit()

	if args.pull:
		pull_task(db, args.pull)
		exit()

	if args.exclude:
		for user_id in args.exclude.split(','):
			exclude_user(db, user_id)
		exit()
