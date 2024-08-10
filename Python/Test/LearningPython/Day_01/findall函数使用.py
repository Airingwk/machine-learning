import re
pattern = '\d\.\d+'
s = 'I study python3.12 every day and python2.11'
s1 = 'I study python every day'
lst = re.findall(pattern, s)
lst1 = re.findall(pattern, s1)
print(lst)
print(lst1)
