import os

print('Success to create python project from template, Nice Job!!\n')

print('Now, this project has below file/directory in root.')
print('====================================================')
filenames = os.listdir('.')
for filename in filenames:
    print(filename)
print('====================================================')
