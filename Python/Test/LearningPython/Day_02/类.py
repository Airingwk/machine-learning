class Student:
    def __init__(self,age,name):
        self.age = age
        self.__name = name

    # 使用 @property 修改方法，将方法转成属性来使用
    @property
    def name(self):
        return self.__name

    # 使用 .setter 设置私有属性为可修改
    @name.setter
    def name(self, Val):
        self.__name = Val

stu = Student(18, 'Dong')
print(stu.age, stu.name)
stu.name = 'Wang'
print(stu.name)