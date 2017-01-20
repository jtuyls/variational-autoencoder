
import matplotlib.pyplot as plt

def visualize_mnist(inputs):
    for i in range(0,len(inputs)):
        image = inputs[i].reshape(28,28)
        plt.imshow(image, cmap='gray')
        plt.savefig('fig_' + str(i) + '.png')
        plt.show()
