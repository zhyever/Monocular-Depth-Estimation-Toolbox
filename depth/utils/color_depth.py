import matplotlib

# color the depth, kitti magma_r, nyu jet
def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # TODO: remove hacks

    # for abs
    # vmin=1e-3
    # vmax=80

    # for relative
    # value[value<=vmin]=vmin

    # vmin=None
    # vmax=None

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :, :3] # bgr -> rgb
    rgb_value = value[..., ::-1]

    return rgb_value