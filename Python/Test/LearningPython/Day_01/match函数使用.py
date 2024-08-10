import re
pattern = '\d\.\d+'  # +限定符， \d 0到9数字 出现一次或多次
s = 'I study python3.14 everyday'
s2 = '3.14 I study python everyday'
match = re.match(pattern, s, re.I)
print(match)
match2 = re.match(pattern, s2, re.I)
print(match2)
print('匹配值的起始位置：', match2.start())
print('匹配值的结束位置：', match2.end())
print('匹配值的区间：', match2.span())
print('匹配区间的位置元素：', match2.span())
print('待匹配的字符串：', match2.string)
print('匹配的数据：', match2.group())