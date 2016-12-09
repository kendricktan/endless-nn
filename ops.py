import os

import numpy as np

# Gets ROI from mouse inputs
def get_roi_from_mouse(clicked_positions):
    '''
    Gets ROI from mouse click positions
    '''
    assert type(clicked_positions) is list
    assert len(clicked_positions) == 2

    return list(
        clicked_positions[0] + tuple(
            np.subtract(clicked_positions[1], clicked_positions[0])
        )
    )


# Check if folder exists
def check_folder_exists(folder='data'):
    if not os.path.exists(folder):
        os.mkdir(folder)


# Get numeric number for new file (e.g. output_{x})
def get_iter_no(folder='data', filename='output_', extension='.csv'):
    iter_name = 0
    while os.path.exists(os.path.join(folder, filename + str(iter_name) + extension)):
        iter_name += 1
    return iter_name


# Gets relative filename for new file (e.g. folders etc joined)
def get_relative_filename(iter_name, folder='data', filename='output_', extension='.csv'):
    fileoutname = filename + str(iter_name) + extension
    fileoutname = os.path.join(folder, fileoutname)
    return fileoutname


# Get latest filename for outfile
def get_new_filename(folder='data', filename='output_', extension='.csv'):
    check_folder_exists(folder)
    iter_name = get_iter_no(folder=folder, filename=filename, extension=extension)
    return get_relative_filename(iter_name, folder=folder, filename=filename, extension=extension)
