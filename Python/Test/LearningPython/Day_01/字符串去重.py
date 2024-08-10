s = 'hhhasyddhdbgshkl'

# 1.字符串拼接+not in
s_new = ''
for item in s:
    if item not in s_new:
        s_new += item
print(s_new)

# 2.下标索引+not in
s_new2 = ''
for i in range(len(s)):
    if s[i] not in s_new2:
        s_new2 += s[i]
print(s_new2)

# 3.集合去重+列表排序
s_new3 = set(s)
lst = list(s_new3)
print(lst)
lst.sort(key=s.index)
print(''.join(lst))






