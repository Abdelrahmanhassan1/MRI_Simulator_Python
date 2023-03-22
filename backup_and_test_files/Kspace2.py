import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

vegetables = ["cucumber", "tomato", "lettuce", "asparagus"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun"]

harvest = np.array([[0.8, 0.3, 0.25, 0.5],
                    [0.75, 0.6, 0.2, 0.4],
                    [1.0, 0.95, 0.3, 0.35],
                    [0.2, 0.3, 0.4, 0.1]])


# [45.254834, 0., 0., 0.],
# [32., 0., 0., 0.],
# [45.254834, 0., 0., 0.]

harvest = np.array([[136., 11.3137085, 8., 11.3137085],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
for i in range(4):

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
    ax.xaxis.tick_top()

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()

    harvest[1, 0] = 45.254834
    harvest[1, 1] = 0
    harvest[1, 2] = 0
    harvest[1, 3] = 0
