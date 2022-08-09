import os
import numpy as np
import imageio
import gym
import pdb

def config_from_png(relpath):
	asset_path = os.path.join(os.path.dirname(__file__), relpath)
	img = imageio.imread(asset_path)
	grid = (img < 100).all(axis=-1)
	config = [
		['x' if i else ' ' for i in row]
		for row in grid
	]
	return config

CONFIGS = {
	'default': [
		'x        x',
		'x        x',
		'xxxx  xxxx',
		'x        x',
		'x        x',
	],
	'snake': [
		'xxxxxxxx  ',
		'xxxxxxxx  ',
		'xxxxxxxx  ',
		'xxxxxxxx  ',
		'          ',
		'          ',
		'  xxxxxxxx',
		'  xxxxxxxx',
		'  xxxxxxxx',
		'  xxxxxxxx',
	],
}

'''
	observation space: [0, 1]^2
	action space: [-1, 1]^2
'''

class MazeEnv(gym.Env):

	def __init__(self, config='default', action_scale=1, max_episode_steps=1000):
		'''
			action_scale : the number of cells an action of 1 maps to
				e.g., action_scale = 5 will cause a continuous action
				of [ 1, 1 ] to move halfway down and halfway across a cell
		'''
		config = CONFIGS[config]
		self._config = np.array(
			[[char for char in row] for row in config]
		)
		self._max_episode_steps = max_episode_steps

		self.height = len(config)
		self.width = len(config[0])
		assert all([len(c) == self.width for c in config]), \
			'All rows of maze should be the same length'

		self.open_cells = np.array([
			(i, j)
			for i in range(self.height)
			for j in range(self.width)
			if config[i][j] == ' '
		])

		self.scales = np.array([1/self.height, 1/self.width])
		self.xlim = (0, self.width)
		self.ylim = (0, self.height)
		self.low  = np.array([self.ylim[0], self.xlim[0]])
		self.high = np.array([self.ylim[1], self.xlim[1]])

		self.action_deltas = np.einsum(
			'i,j->ij',
			np.linspace(0, action_scale, 20),
			self.scales,
		)

	def _cell_to_float(self, cells, stochastic=True):
		'''
			cell corresponds to upper left location by default;
			adding noise (scaled by `self.scales`) pushes the location
			down and to the right within the same cell
		'''
		if stochastic:
			offsets = np.random.uniform(low=0, high=1, size=cells.shape)
			cells = cells + offsets
		locations = cells * self.scales
		return locations

	def _float_to_cell(self, locations):
		cells = np.floor(locations / self.scales)
		return cells.astype(np.int32)

	def _is_open(self, locations):
		cells = self._float_to_cell(locations)
		I = cells[:,:,0]
		J = cells[:,:,1]
		symbols = self._config[I.reshape(-1), J.reshape(-1)]
		mask = (symbols == ' ').reshape(I.shape)
		return mask

	# def reset(self):
	# 	return self._vec_reset(1)

	def _get_obs(self):
		return self._pos.copy()

	# def step(self, action):
	# 	pass

	def vec_reset(self, N):
		n_open = len(self.open_cells)
		inds = np.random.choice(n_open, size=N)
		cells = self.open_cells[inds]
		self._pos = self._cell_to_float(cells)
		self.vec_N = N
		return self._get_obs()

	def vec_step(self, actions, eps=1e-6):
		'''
			actions : [ N x 2 ]
				(∆i, ∆j)
		'''
		assert len(actions) == self.vec_N
		candidate_deltas = np.einsum('jd,nd->jnd', self.action_deltas, actions)
		## [ n_steps x N x 2 ] = [ N x 2 ] + [ n_steps x N x 2 ] * [ ]
		candidates = self._pos + candidate_deltas
		candidates = np.clip(candidates, 0, 1-eps)
		valid = self._is_open(candidates)

		## once a sequence is out of bounds, it cannot go back in bounds
		valid = np.cumprod(valid, axis=0)

		## the largest index for which `valid` is True along axis 0
		argmax = np.einsum('sn,s->sn', valid, np.arange(len(valid))).argmax(0)
		self._pos = candidates[argmax, np.arange(self.vec_N)]

		rew = np.zeros(self.vec_N)
		term = np.array([False for _ in range(self.vec_N)])
		return self._get_obs(), rew, term, {}


if __name__ == '__main__':
	N = 4
	env = MazeEnv() #'small'

	obs = env.vec_reset(N)
	actions = np.random.uniform(-1, 1, size=(N, 2))
	next_obs, rew, term, _ = env.vec_step(actions)

	pdb.set_trace()