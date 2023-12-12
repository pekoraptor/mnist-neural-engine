import matplotlib.pyplot as plt
import numpy as np

def plt_function3D(f, additionalPoints=None, pointsColors=None,
                   pltCmap='PiYG', visibility=0.3):
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = np.zeros_like(x1)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            y[i, j] = f([x1[i, j], x2[i, j]])

    figure = plt.figure()
    plot = figure.add_subplot(111, projection='3d')
    plot.plot_surface(x1, x2, y, cmap=pltCmap, alpha=visibility)

    if additionalPoints:
        for index, setOfPoints in enumerate(additionalPoints):
            additionalX1, additionalX2 = setOfPoints
            plot.scatter(additionalX1, additionalX2, f(setOfPoints),
                         c=pointsColors[index % len(pointsColors)],
                         s=100)

    plot.set_xlabel('x1')
    plot.set_ylabel('x2')
    plot.set_zlabel('g(x1, x2)')
    plot.set_title('3D Plot of g(x1, x2)')

    plt.legend()
    plt.show()

# if __name__ == "__main__":
#     plt_function3D(g, sgd(g, gGradient, [1, 1.7], 0.5, 10000, 0.01, 4, 1)[1], ['red', 'blue', 'green'])