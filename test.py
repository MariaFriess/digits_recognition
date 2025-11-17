class calc():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def plus(self):
        return self.a + self.b

    def minus(self):
        return self.a - self.b


inp_arr = input('Enter two numbers: ').split(' ')
c = calc(int(inp_arr[0]), int(inp_arr[1]))

res = c.plus()
print(res)
