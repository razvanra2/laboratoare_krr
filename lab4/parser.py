from pprint import pprint
from copy import deepcopy


class Parser(object):
	@staticmethod
	def all_descendants(graph, bn, x):
		desc = [x]
		for c in graph[x]:
			desc.append(Parser.all_descendants(graph, bn, c))
		return [item for sublist in desc for item in sublist]

	@staticmethod
	def parse(file: str):
		'''
		@param file: path to the input file
		:returns Bayesian network as a dictionary {node: [list of parents], ...}
		and the list of queries as [{"X": [list of vars],
		"Y": [list of vars], "Z": [list of vars]}, ... ] where we want
		to test the conditional independence of vars1 ⊥ vars2 | cond
		'''
		bn = {}
		queries = []

		with open(file) as fin:
			# read the number of vars involved
			# and the number of queries
			N, M = [int(x) for x in next(fin).split()]

			# read the vars and their parents
			for i in range(N):
				line = next(fin).split()
				var, parents = line[0], line[1:]
				bn[var] = parents

			# read the queries
			for i in range(M):
				vars, cond = next(fin).split('|')

				# parse vars
				X, Y = vars.split(';')
				X = X.split()
				Y = Y.split()

				# parse cond
				Z = cond.split()

				queries.append({
					"X": X,
					"Y": Y,
					"Z": Z
				})

			# read the answers
			for i in range(M):
				queries[i]["answer"] = next(fin).strip()

		return bn, queries

	@staticmethod
	def get_graph(bn: dict):
		'''
		@param bn: Bayesian network obtained from parse
		:returns the graph as {node: [list of children], ...}
		'''
		graph = {}

		for node in bn:
			parents = bn[node]

			# this is for the leafs
			if node not in graph:
				graph[node] = []

			# for each parent add
			# the edge parent->node
			for p in parents:
				if p not in graph:
					graph[p] = []
				graph[p].append(node)

		return graph

	@staticmethod
	def test_causal(graph, bn, x, Y, Z, visited):
		children = list(c for c in graph[x] if c not in visited)

		for z in children:
			for y in graph[z]:
				if z not in Z and y in Y:
					return False
			if not Parser.test_independence(graph, bn, z, Y, Z, visited):
				return False
		return True


	@staticmethod
	def test_evidential(graph, bn, x, Y, Z, visited):
		parents = list(p for p in bn[x] if p not in visited)

		for z in parents:
			for y in bn[z]:
				if z not in Z and y in Y:
					return False
			if not Parser.test_independence(graph, bn, z, Y, Z, visited):
				return False
		return True

	@staticmethod
	def test_cause(graph, bn, x, Y, Z, visited):
		parents = list(p for p in bn[x] if p not in visited)

		for z in parents:
			for y in graph[z]:
				if z not in Z and y in Y:
					return False
			if not Parser.test_independence(graph, bn, z, Y, Z, visited):
				return False
		return True

	@staticmethod
	def test_effect(graph, bn, x, Y, Z, visited):
		children = list(c for c in graph[x] if c not in visited)

		for z in children:
			for y in list(filter(lambda d: d in bn[z], Y)):
				descendants = Parser.all_descendants(graph, bn, z)
				for desc in descendants:
					if desc in Z:
						return False
			if not Parser.test_independence(graph, bn, z, Y, Z, visited):
				return False
		return True

	@staticmethod
	def test_independence(graph, bn, X, Y, Z, visited=[]):
		for x in X:
			visited.append(x)
			if not(Parser.test_causal(graph, bn, x, Y, Z, deepcopy(visited))) or not(Parser.test_evidential(graph, bn, x, Y, Z, deepcopy(visited))) or not(Parser.test_cause(graph, bn, x, Y, Z, deepcopy(visited))) or not(Parser.test_effect(graph, bn, x, Y, Z, deepcopy(visited))):
				return False
		return True


if __name__ == "__main__":
	bn, queries = Parser.parse("./bn1")
	graph = Parser.get_graph(bn)

	for q in queries:
		print(q)
		X = q['X']
		Y = q['Y']
		Z = q['Z']
		print(Parser.test_independence(graph, bn, X, Y, Z))

