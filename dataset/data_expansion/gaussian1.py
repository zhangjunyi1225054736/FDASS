import cv2




kernel_size_1 = (7,7)
kernel_size_2 = (9,9)

sigma1 = 1.5
sigma2 = 2.0
sigma3 = 3.5
sigma4 = 5


f = open('train_1.txt')

while 1 :
  line = f.readline()
  if not line:
     break;
 #print(line.strip())


  name = line.strip().split('/')[1]
  fold = line.strip().split('/')[0]
  url = line.strip()
  #img = cv2.imread('/home/zhangjunyi/AdaptSegNet/data/cityscapes/leftImg8bit/train/'+url);

  #root = '/home/zhangjunyi/AdaptSegNet/data/cityscapes/leftImg8bit/train/' + fold + '/'
  #print(root)
  new_imgName_1 = fold+'/new_1_' + name
  new_imgName_2 = fold+'/new_2_' + name
  new_imgName_3 = fold+'/new_3_' + name
  new_imgName_4 = fold+'/new_4_' + name
  new_imgName_5 = fold+'/new_5_' + name
  new_imgName_6 = fold+'/new_6_' + name
  new_imgName_7 = fold+'/new_7_' + name
  new_imgName_8 = fold+'/new_8_' + name
  
  print(new_imgName_1)
  print(new_imgName_2)
  print(new_imgName_3)
  print(new_imgName_4)
  print(new_imgName_5)
  print(new_imgName_6)
  print(new_imgName_7)
  print(new_imgName_8)
  
 
