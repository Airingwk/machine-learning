import re
pattern = '\d\.\d+'
s = 'I love python3.12 every day and python3.3 as well'
s1 = 'I study python every day'
match = re.search(pattern, s)
match2 = re.search(pattern, s1)
print(match)
print(match2)
print(match.group())

