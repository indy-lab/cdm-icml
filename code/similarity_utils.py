import csv
import re

def read_data(dataset):
	filename = '../data/similarity/'+dataset+'.csv'
	data_dict = {}
	with open(filename, mode='r') as infile:
		reader = csv.DictReader(infile)
		for row in reader:
			for k, v in row.items():
				if v != '':
					try:
						data_dict.setdefault(k,[]).append(int(v))
					except:
						data_dict.setdefault(k,[]).append(v)
				else:
					data_dict.setdefault(k,[]).append(None)
	return data_dict

def feature_map(input):
	"""
	Maps tuples of variables into a unique index

	Features are sometimes more than one variable, but for the purpose of
	embeddings, it is best to index them by one variable - this maps collections of
	variables to an index. Ignores None values
	Inputs: input - a list of lists where each list is a feature over all data.
	The code forms tuples from each lists, and adds a mapping to an index, and 
	returns that mapping as a dict. The lists must be of same size

	Outputs: dictionary containing mapping.

	"""
	data_map = {};
	map_idx = 0
	data_len = len(input[0])
	for data_list in input:
		assert data_len == len(data_list)

	for data_idx in range(data_len):
		datapoint = ()
		for idx in range(len(input)):
			datapoint += (input[idx][data_idx],)
		
		if datapoint not in data_map and not any(
				map(lambda x: x is None, datapoint)):
			data_map[datapoint] = map_idx
			map_idx += 1

	return data_map

def feature_id(data_dict, features, context_size=None, cluster=False, k=50):
	"""
	Adds adds feature_ids to the data in data_dict.

	This maps collections of variables in a list to an index, making it useful for
	embeddings. Maps None values to the largest index, incremented (num_features)
	Inputs: data_dict - dict containing data, with features labled by headers
	features - list of headers eg. ['x','p'] that contain feature types
	context_size (optional) - Can limit max size. 
	Size of the choice context, so it sweeps for all the features in the data

	Outputs: data_map, the mapping between ids and features
	         (also modifies data_dict in place)
	"""
	# Figure out max choice set length:
	if context_size is None:
		matches = [re.match(features[0]+r'(\d+)', k) for k in data_dict.keys()]
		context_size = max([int(m.group(1)) for m in matches if m])
	temp_list = []
	ismultiset = False
	for i in range(len(features)):
		temp_list.append([])
		for j in range(1,context_size+1):
			temp_list[-1] += data_dict[features[i]+str(j)]
	data_map = feature_map(temp_list)
	# if cluster:
	# 	data_map = cluster_features(data_map,k)
	num_features = len(data_map.values())

	# data_dict['context_ids'] = []
	# data_dict['choice_id'] = []
	# data_dict['context_ids_wo_choice'] = []
	# data_dict['choice_set_lengths'] = []
	# for j in range(len(data_dict[features[0]+'1'])):
	# 	data_dict['context_ids'].append([])
	# 	data_dict['context_ids_wo_choice'].append([])
	# 	counter = 0
	# 	for i in range(1,context_size+1):
	# 		datapoint = ()
	# 		for k in range(len(features)):
	# 			datapoint += (data_dict[features[k]+str(i)][j],)
	# 		if any(map(lambda x: x is None, datapoint)):
	# 			ismultiset=True
	# 			data_dict['context_ids'][-1].append(num_features)
	# 			data_dict['context_ids_wo_choice'][-1].append(num_features)
	# 		else:
	# 			counter += 1
	# 			data_dict['context_ids'][-1].append(data_map[datapoint])
	# 			data_dict['context_ids_wo_choice'][-1].append(
	# 														data_map[datapoint])
	# 	chosen_slot_key = handle_chosen_slot(data_dict)
	# 	data_dict['choice_id'].append(data_dict['context_ids'][-1][data_dict[chosen_slot_key][j]])
	# 	del data_dict['context_ids_wo_choice'][-1][data_dict[chosen_slot_key][j]]
	# 	data_dict['choice_set_lengths'].append(counter)
	
	return data_map, context_size, ismultiset
