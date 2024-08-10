class Father01():
    def __init__(self,name):
        self.name = name

    def show01(self):
        print('Father01 中的方法调用')

class Father02():
    def __init__(self,age):
        self.age = age

    def show02(self):
        print('Father02 中的方法调用')

class Son(Father01,Father02):
    def __init__(self,name,age,gender):
        Father01.__init__(self,name)
        Father02.__init__(self,age)
        self.gender = gender

son = Son('Wang', 20,'male')
son.show01()
son.show02()