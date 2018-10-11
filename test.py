from model import *

if __name__ == "__main__":
    x = Variable(torch.randn((2, 3)))
    hw = HighWay(2, 3, F.relu)

    print(hw(x))

    x = Variable(torch.randn((2, 3)))
    hw = HighWay(1, 3, F.tanh)

    print(hw(x))