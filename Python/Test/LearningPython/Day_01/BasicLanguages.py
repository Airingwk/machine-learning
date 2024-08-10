import random

print(list(random.randint(1, 20) for i in range(0, 10)))

lst = [20, "t", "yyfgt", '你好']
print(lst)
lst.append('20rghy')
print(lst)
a = lst.pop(2)
print(a)
print(lst)
lst.insert(-2, 100)
print(lst)
lst[0] = 100
print(lst)




