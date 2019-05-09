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
  img = cv2.imread('/home/zhangjunyi/AdaptSegNet/data/cityscapes/leftImg8bit/train/'+url);
  img_1 = cv2.GaussianBlur(img,kernel_size_1,sigma1)
  img_2 = cv2.GaussianBlur(img,kernel_size_1,sigma2)
  img_3 = cv2.GaussianBlur(img,kernel_size_1,sigma3)
  img_4 = cv2.GaussianBlur(img,kernel_size_1,sigma4)
  img_5 = cv2.GaussianBlur(img,kernel_size_2,sigma1)
  img_6 = cv2.GaussianBlur(img,kernel_size_2,sigma2)
  img_7 = cv2.GaussianBlur(img,kernel_size_2,sigma3)
  img_8 = cv2.GaussianBlur(img,kernel_size_2,sigma4)

  root = '/home/zhangjunyi/AdaptSegNet/data/cityscapes/leftImg8bit/train/' + fold + '/'
  print(root)
  new_imgName_1 = 'new_1_' + name
  new_imgName_2 = 'new_2_' + name
  new_imgName_3 = 'new_3_' + name
  new_imgName_4 = 'new_4_' + name
  new_imgName_5 = 'new_5_' + name
  new_imgName_6 = 'new_6_' + name
  new_imgName_7 = 'new_7_' + name
  new_imgName_8 = 'new_8_' + name

  cv2.imwrite(root+new_imgName_1,img_1) 
  cv2.imwrite(root+new_imgName_2,img_2) 
  cv2.imwrite(root+new_imgName_3,img_3) 
  cv2.imwrite(root+new_imgName_4,img_4) 
  cv2.imwrite(root+new_imgName_5,img_5) 
  cv2.imwrite(root+new_imgName_6,img_6) 
  cv2.imwrite(root+new_imgName_7,img_7) 
  cv2.imwrite(root+new_imgName_8,img_8) 
 
