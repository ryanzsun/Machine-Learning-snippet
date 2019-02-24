import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CART():
	def __init__(self, max_depth, min_size):
		self.max_depth = max_depth
		self.min_size = min_size
		self.root = None


	def build_tree(self, data):
		root = self.find_best_split(data)
		self.split(root, 1)
		return root


	def find_best_split(self, data):
		label = data[:,-1]
		best_split_feature, best_split_value, best_gini, split_data = 1000, 1000, 1000, None

		for i in range(data.shape[1]-1):
			split_values = np.unique(data[:,i])

			for value in split_values:
				true_part = data[data[:,i] < value]
				false_part = data[data[:,i] >= value]

				gini = self.gini_index(true_part, false_part)
				# print(gini)
				if gini < best_gini:
					best_gini = gini
					best_split_value = value
					best_split_feature = i
					split_data = [true_part, false_part]

		return {'feature':best_split_feature, 'value':best_split_value, "split_data":split_data}


	def split(self, node, depth):
		left, right = node['split_data'][0], node['split_data'][1]

		if left.shape[0] == 0 or right.shape[0] == 0:
			node['left'] = node['right'] = self.leaf(np.concatenate((left,right), axis = 0))
			return
		if depth >= self.max_depth:
			node['left'],node['right'] = self.leaf(left), self.leaf(right)
			return

		if left.shape[0] <= self.min_size:
			node['left'] = self.leaf(left)
		else:
			node['left'] = self.find_best_split(left)
			self.split(node['left'], depth + 1)

		if right.shape[0] <= self.min_size:
			node['right'] = self.leaf(right)
		else:
			node['right'] = self.find_best_split(right)
			self.split(node['right'], depth + 1)


	def leaf(self, data):
		labels = np.array(data[:,-1].astype(int))
		return np.argmax(np.bincount(labels))


	def gini_index(self, true_part, false_part):
		t_labels, f_labels = np.unique(true_part[:,-1]), np.unique(false_part[:,-1])
		true_gini_prob, false_gini_prob = 1., 1.
		for l in t_labels:
			true_gini_prob = true_gini_prob - (float(true_part[true_part[:,-1] == l].shape[0]) / float(true_part.shape[0]))**2
		for l in f_labels:
			false_gini_prob = false_gini_prob - (float(false_part[false_part[:,-1] == l].shape[0]) / float(false_part.shape[0]))**2
		# print(true_gini_prob, false_gini_prob)
		total_samples = float(true_part.shape[0] + false_part.shape[0])
		return (float(true_part.shape[0]) / total_samples) * true_gini_prob + \
					(float(false_part.shape[0]) / total_samples) * false_gini_prob

	def fit(self, data, label):
		self.root = self.build_tree(np.concatenate((X_train, y_train), axis=1))

	def predict(self, data):
		result = np.zeros((data.shape[0]))
		for i in range(data.shape[0]):
			node = self.root
			result[i] = self.predict_sample(node, data[i,:])
		return result

	def predict_sample(self, node, data):
		if data[node['feature']] < node['value']:
			if isinstance(node['left'], dict):
				return self.predict_sample(node['left'], data)
			else:
				return node['left'] 
		else:
			if isinstance(node['right'], dict):
				return self.predict_sample(node['right'], data)
			else:
				return node['right'] 

if __name__ == "__main__":
	data = load_wine()

	X_train, X_test, y_train, y_test = train_test_split(
		data.data, data.target, test_size=0.33, random_state=42)

	y_train = np.expand_dims(y_train, axis = 1)
	classifier = CART(max_depth = 10, min_size = 3)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	print(accuracy_score(y_test, y_pred))
