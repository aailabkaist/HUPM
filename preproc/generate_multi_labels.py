import numpy as np
import os
import argparse

dataset = 'cars'

# images: ['train/Cadillac_Escalade_EXT_Crew_Cab_2007/00056.jpg',  'train/Cadillac_Escalade_EXT_Crew_Cab_2007/00124.jpg', ..]
image_paths = []
np.save(os.path.join('data', dataset, 'formatted_train_images.npy'), image_paths)

# labels: (# of data * # of classes)
multi_labels = []
np.save(os.path.join('data', dataset, 'formatted_train_labels.npy'), multi_labels)
