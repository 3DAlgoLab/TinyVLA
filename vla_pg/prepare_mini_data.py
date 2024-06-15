# %%
from icecream import ic
import numpy as np

# Language Table Simulation based dataset
START_ID = 0
MAX_STEP_RANGE = 60  # mm (-30 ~ + 30)


def trans_val_to_id(action_val):
    action_val = np.clip(action_val, -0.03, 0.03)
    return int((action_val * 1000.0 + MAX_STEP_RANGE / 2) * 255 / 60 + 0.5) + START_ID


def action_id_to_trans_val(action_id):
    action_id = np.clip(action_id, 0, 255)
    return (action_id - 128) * MAX_STEP_RANGE / 255 / 1000.0


ic(action_id_to_trans_val(128))
ic(action_id_to_trans_val(256))
ic(action_id_to_trans_val(0))

ic(trans_val_to_id(0))
ic(trans_val_to_id(0.030))
ic(trans_val_to_id(-0.030))
