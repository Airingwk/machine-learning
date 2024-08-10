import re
pattern = '黑客|反爬|破解'
s = '我想学习一些python，破解VIP视频，学黑客进行反爬操作'
s_new = re.sub(pattern, 'XXX', s)
print(s_new)

s1 = 'https://www.baidu.com/s?wd=ysj&rsv_spt=1&rsv_iqid=0xadf820f700035743'
pattern1 = '[?|&]'
s1_new = re.split(pattern1, s1)
print(s1_new)

