"""Pre-registered prompt registry for the anchor library (2026-08-31).

K_train = 16 anchors, Q_heldout = 8 prompts never seen in training.
Train prompts follow GenRL's DOMAIN2PREDICATES style for the walker domain
(descriptors + direction/speed variants). Held-out prompts are novel
descriptors not present in the train set.

FROZEN: these lists are part of the pre-registration. Do not edit after the
M0 preflight passes — changing them invalidates every downstream gate.
"""

TRAIN_PROMPTS = [
    # GenRL walker predicate set (descriptors)
    'taking a walk',
    'running fast',
    'standing up vertically on both feet',
    'single-leg balancing',
    'standing upside down',
    'high kick',
    'stepping forward',
    'standing on one bended knee',
    'lying down on the back with one raised leg',
    'sitting on the knees',
    'dog yoga pose',
    'lying down horizontally',
    'walking',
    # direction / speed variants
    'walking backwards',
    'running backwards',
    'walking slowly',
]

HELDOUT_PROMPTS = [
    'crouching down',
    'jumping in place',
    'marching in place',
    'kneeling down',
    'limping',
    'walking with arms swinging',
    'spinning around',
    'stretching arms forward',
]

K_TRAIN = len(TRAIN_PROMPTS)      # 16
K_HELDOUT = len(HELDOUT_PROMPTS)  # 8
assert K_TRAIN == 16 and K_HELDOUT == 8

# Fixed derangement for the shuffled-condition control (perm[i] != i for all i;
# verified: 5 shares no factor with 16).
SHUFFLE_PERM = [(i + 5) % 16 for i in range(16)]
assert all(SHUFFLE_PERM[i] != i for i in range(16))
