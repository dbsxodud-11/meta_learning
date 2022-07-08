import numpy as np

def context_target_split(x, y, num_context, num_extra_target):

    num_points = x.shape[1]
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target