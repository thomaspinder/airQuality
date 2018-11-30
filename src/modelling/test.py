def adder(**kwargs):
    print(kwargs['a'])


if __name__ == '__main__':
    adds = {'a':1, 'b':2, 'c':3}
    adder(a=1, b=2)