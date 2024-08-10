s1 = 'hello'
s2 = 'world'

print(s1+s2)
print(''.join([s1, s2]))  # 使用空字符进行拼接
print('*'.join(['hello', 'cat', 'doge', 'maomi']))
print('hello''world')
print('%s%s' % (s1, s2))
print('{0}{1}'.format(s1, s2))
print(f'{s1}{s2}')
