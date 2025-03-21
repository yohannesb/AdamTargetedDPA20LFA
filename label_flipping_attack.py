from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.utils import default_no_change
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

if __name__ == '__main__':
    START_EXP_IDX = 3000
    NUM_EXP = 10
    
    # START_EXP_IDX = 3000
    # NUM_EXP = 10
    # NUM_POISONED_WORKERS = 5
    NUM_POISONED_WORKERS = 25
    REPLACEMENT_METHOD = default_no_change
    KWARGS = {
        #"NUM_WORKERS_PER_ROUND" : 1
        "NUM_WORKERS_PER_ROUND" : 5
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
