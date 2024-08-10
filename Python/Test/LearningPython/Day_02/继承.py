class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print(f'大家好，我叫:{self.name}, 今年:{self.age}岁')

# Student类 继承 Person类
class Student(Person):
    def __init__(self, name, age, stuNum):
        super().__init__(name, age) # 调用父类的初始化信息
        self.stuNum = stuNum

# Doctor类 继承 Person类
class Doctor(Person):
    def __init__(self, name, age, department):
        super().__init__(name, age)
        self.department = department


stu = Student('Dong', 18,'24001')
stu.show()

doctor = Doctor('Wang',30,'Surgery')
doctor.show()



