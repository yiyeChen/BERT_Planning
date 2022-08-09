import pdb

PAD, MASK = '<-PAD->', '<-MASK->'
RIGHT = '<-RIGHT->' #[1,0]
LEFT = '<-LEFT->' #[-1,0]
UP = '<-UP->' #[0,-1]
DOWN = '<-DOWN->' #[0,1]

tokens = [PAD, MASK, RIGHT, LEFT, UP, DOWN]

_token2idx = dict(zip(tokens, range(len(tokens))))

pdb.set_trace()