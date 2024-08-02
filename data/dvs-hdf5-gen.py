from dcll.dcll.pytorch_libdcll import *
from dcll.dcll.experiment_tools import *
from dcll.dcll.load_dvsgestures_sparse import *
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1111)


filename = os.path.join(dcll_folder, '../../data/DvsGesture/dvs_gestures_events.hdf5')

create_events_hdf5(filename)

