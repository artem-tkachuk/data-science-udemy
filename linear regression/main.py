from train import train
from graph import init_line
import matplotlib.pyplot as plt

def main():
    fileName = 'data/cost_revenue_clean_custom.txt'

    thetas = train(fileName)
    print(thetas)
    plt.show()


if __name__ == '__main__':
    main()