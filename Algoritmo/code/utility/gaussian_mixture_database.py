import numpy as np


def generate_gauss_dataset():
    n_samples = 300
    mu1, sigma1 = np.array([0, 0]), 5  # mean and variance
    mu2, sigma2 = np.array([5, -4]), 5
    mu3, sigma3 = np.array([11, 5]), 5

    x1 = np.random.normal(mu1, np.sqrt(sigma1), (n_samples, 2))
    x2 = np.random.normal(mu2, np.sqrt(sigma2), (n_samples, 2))
    x3 = np.random.normal(mu3, np.sqrt(sigma3), (n_samples, 2))

    x4 = np.zeros((n_samples, 1), float)
    x4.fill(0)
    x1 = np.append(x1, x4, axis=1)
    x2 = np.append(x2, x4, axis=1)
    x4.fill(1)
    x3 = np.append(x3, x4, axis=1)

    X = np.append(x1, x2, axis=0)
    X = np.append(X, x3, axis=0)

    np.savetxt("gauss_data_1.csv", X, delimiter=",")


def generate_gauss_dataset2():
    n_samples = 120
    mu1, sigma1 = np.array([0, 0]), 3  # mean and variance
    mu2, sigma2 = np.array([6, -4]), 3
    mu3, sigma3 = np.array([11, 5]), 3

    x1 = np.random.normal(mu1, np.sqrt(sigma1), (n_samples, 2))
    x2 = np.random.normal(mu2, np.sqrt(sigma2), (n_samples, 2))
    x3 = np.random.normal(mu3, np.sqrt(sigma3), (n_samples, 2))

    x4 = np.zeros((n_samples, 1), float)
    x4.fill(0)
    x1 = np.append(x1, x4, axis=1)
    x4.fill(1)
    x2 = np.append(x2, x4, axis=1)
    x4.fill(2)
    x3 = np.append(x3, x4, axis=1)

    X = np.append(x1, x2, axis=0)
    X = np.append(X, x3, axis=0)
    np.random.shuffle(X)

    marker = 'o'
    size = 50
    alpha = 1
    color = ['b', 'r', 'y', 'g']
    import matplotlib.pyplot as plt
    f = plt.figure(0, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    classes = list(set(X[:, -1]))
    for i, c in enumerate(classes):
        data = X[X[:, -1] == c]
        plt.scatter(data[:, 0:1], data[:, 1:2], marker=marker, color=color[i], s=size, edgecolors='black',
                    linewidths=1.5,
                    alpha=alpha, label="Datapoints")
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title("Gauss 2")
    plt.show()
    plt.clf()
    # plt.savefig(path_result + "/" + namefile_result + ".png")
    X_tot = np.copy(X)

    n_samples =90
    mu1, sigma1 = np.array([0, 0]), 3  # mean and variance
    mu2, sigma2 = np.array([6, -4]), 3
    mu3, sigma3 = np.array([11, 5]), 3
    mu4, sigma4 = np.array([16, -3]), 3

    x1 = np.random.normal(mu1, np.sqrt(sigma1), (n_samples, 2))
    x2 = np.random.normal(mu2, np.sqrt(sigma2), (n_samples, 2))
    x3 = np.random.normal(mu3, np.sqrt(sigma3), (n_samples, 2))
    x4 = np.random.normal(mu4, np.sqrt(sigma4), (n_samples, 2))

    x5 = np.zeros((n_samples, 1), float)
    x5.fill(0)
    x1 = np.append(x1, x5, axis=1)
    x5.fill(1)
    x2 = np.append(x2, x5, axis=1)
    x5.fill(2)
    x3 = np.append(x3, x5, axis=1)
    x5.fill(3)
    x4 = np.append(x4, x5, axis=1)

    X = np.append(x1, x2, axis=0)
    X = np.append(X, x3, axis=0)
    X = np.append(X, x4, axis=0)

    np.random.shuffle(X)
    marker = 'o'
    size = 50
    alpha = 1
    color = ['b', 'r', 'y', 'g']
    # import matplotlib.pyplot as plt
    f = plt.figure(0, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    classes = list(set(X[:, -1]))
    for i, c in enumerate(classes):
        data = X[X[:, -1] == c]
        plt.scatter(data[:, 0:1], data[:, 1:2], marker=marker, color=color[i], s=size, edgecolors='black',
                    linewidths=1.5,
                    alpha=alpha, label="Datapoints")
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title("Gauss 2")
    plt.show()
    plt.clf()

    X_tot = np.append(X_tot,X,axis=0)

    np.savetxt("gauss_data_2.csv", X_tot, delimiter=",")

def generate_gauss_dataset3():
    n_samples = 180
    mu1, sigma1 = np.array([0, 0]), 3  # mean and variance
    mu2, sigma2 = np.array([6, -4]), 3
    # mu3, sigma3 = np.array([11, 5]), 5

    x1 = np.random.normal(mu1, np.sqrt(sigma1), (n_samples, 2))
    x2 = np.random.normal(mu2, np.sqrt(sigma2), (n_samples, 2))

    x4 = np.zeros((n_samples, 1), float)
    x4.fill(0)
    x1 = np.append(x1, x4, axis=1)
    x4.fill(1)
    x2 = np.append(x2, x4, axis=1)

    X = np.append(x1, x2, axis=0)
    np.random.shuffle(X)

    marker = 'o'
    size = 50
    alpha = 1
    color = ['b', 'r', 'y', 'g']
    import matplotlib.pyplot as plt
    f = plt.figure(0, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    classes = list(set(X[:, -1]))
    for i, c in enumerate(classes):
        data = X[X[:, -1] == c]
        plt.scatter(data[:, 0:1], data[:, 1:2], marker=marker, color=color[i], s=size, edgecolors='black',
                    linewidths=1.5,
                    alpha=alpha, label="Datapoints")
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title("Gauss 2")
    plt.show()
    plt.clf()
    # plt.savefig(path_result + "/" + namefile_result + ".png")
    X_tot = np.copy(X)

    n_samples = 120
    mu1, sigma1 = np.array([0, 0]), 3  # mean and variance
    mu2, sigma2 = np.array([6, -4]), 3
    mu3, sigma3 = np.array([11, 5]), 3

    x1 = np.random.normal(mu1, np.sqrt(sigma1), (n_samples, 2))
    x2 = np.random.normal(mu2, np.sqrt(sigma2), (n_samples, 2))
    x3 = np.random.normal(mu3, np.sqrt(sigma3), (n_samples, 2))

    x4 = np.zeros((n_samples, 1), float)
    x4.fill(0)
    x1 = np.append(x1, x4, axis=1)
    x4.fill(1)
    x2 = np.append(x2, x4, axis=1)
    x4.fill(2)
    x3 = np.append(x3, x4, axis=1)

    X = np.append(x1, x2, axis=0)
    X = np.append(X, x3, axis=0)
    np.random.shuffle(X)

    marker = 'o'
    size = 50
    alpha = 1
    color = ['b', 'r', 'y', 'g']
    import matplotlib.pyplot as plt
    f = plt.figure(0, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    classes = list(set(X[:, -1]))
    for i, c in enumerate(classes):
        data = X[X[:, -1] == c]
        plt.scatter(data[:, 0:1], data[:, 1:2], marker=marker, color=color[i], s=size, edgecolors='black',
                    linewidths=1.5,
                    alpha=alpha, label="Datapoints")
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title("Gauss 2")
    plt.show()
    plt.clf()
    X_tot = np.append(X_tot, X, axis=0)

    n_samples =90
    mu1, sigma1 = np.array([0, 0]), 3  # mean and variance
    mu2, sigma2 = np.array([6, -4]), 3
    mu3, sigma3 = np.array([11, 5]), 3
    mu4, sigma4 = np.array([16, -3]), 3

    x1 = np.random.normal(mu1, np.sqrt(sigma1), (n_samples, 2))
    x2 = np.random.normal(mu2, np.sqrt(sigma2), (n_samples, 2))
    x3 = np.random.normal(mu3, np.sqrt(sigma3), (n_samples, 2))
    x4 = np.random.normal(mu4, np.sqrt(sigma4), (n_samples, 2))

    x5 = np.zeros((n_samples, 1), float)
    x5.fill(0)
    x1 = np.append(x1, x5, axis=1)
    x5.fill(1)
    x2 = np.append(x2, x5, axis=1)
    x5.fill(2)
    x3 = np.append(x3, x5, axis=1)
    x5.fill(3)
    x4 = np.append(x4, x5, axis=1)

    X = np.append(x1, x2, axis=0)
    X = np.append(X, x3, axis=0)
    X = np.append(X, x4, axis=0)

    np.random.shuffle(X)
    marker = 'o'
    size = 50
    alpha = 1
    color = ['b', 'r', 'y', 'g']
    # import matplotlib.pyplot as plt
    f = plt.figure(0, figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    classes = list(set(X[:, -1]))
    for i, c in enumerate(classes):
        data = X[X[:, -1] == c]
        plt.scatter(data[:, 0:1], data[:, 1:2], marker=marker, color=color[i], s=size, edgecolors='black',
                    linewidths=1.5,
                    alpha=alpha, label="Datapoints")
    plt.legend(bbox_to_anchor=(1.3, 0.5), loc="lower right", labelspacing=2)
    plt.tight_layout()
    plt.title("Gauss 2")
    plt.show()
    plt.clf()

    X_tot = np.append(X_tot,X,axis=0)

    np.savetxt("gauss_data_3.csv", X_tot, delimiter=",")

generate_gauss_dataset3()
