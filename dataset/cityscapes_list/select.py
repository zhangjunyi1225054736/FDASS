file = './train.txt'

f = open(file)
i = 0
while 1:
  line = f.readline()
  if i % 25 ==0:
     print(line.strip())
  i = i + 1
  if not line:
     break
