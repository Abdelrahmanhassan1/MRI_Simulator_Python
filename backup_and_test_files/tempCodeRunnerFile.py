phases = Gy.reshape(-1, 1) * np.arange(rows).reshape(1, -1)[:, :, None] \
    #     + Gx.reshape(-1, 1) * np.arange(columns).reshape(1, 1, -1)