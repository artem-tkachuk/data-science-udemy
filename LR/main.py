from LR.train import train
from LR.graph import init_line
import matplotlib.pyplot as plt

def main():
    fileName = 'cost_revenue_clean_custom.txt'

    thetas = train(fileName)
    print(thetas)
    plt.show()


if __name__ == '__main__':
    main()