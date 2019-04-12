import os, shutil
# 原始数据集解压目录的路径
original_dataset_dir = 'dataSet/orgin_data'
# 保存较小数据集的目录
base_dir = 'dataSet/cats_and_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
#（以下5行）分别对应划分后的训练、验证和测试的目录
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
#（以下2行）猫的训练图像目录
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
#（以下2行）狗的训练图像目录
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
#（以下2行）猫的验证图像目录
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#（以下2行）狗的测试图像目录
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
#（以下2行）猫的测试图像目录
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
#（以下2行）狗的测试图像目录
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
#（以下5行）将前1000张猫的图像复制到train_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
#（以下5行）将接下来500张猫的图像复制到validation_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
#（以下5行）将接下来的500张猫的图像复制到test_cats_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
#（以下5行）将前1000张狗的图像复制到train_dogs_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
#（以下5行）将接下来500张狗的图像复制到validation_dogs_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
#（以下5行）将接下来500张狗的图像复制到test_dogs_dir
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)