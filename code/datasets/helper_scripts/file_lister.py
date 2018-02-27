import os

for file in os.listdir('.'):
	if str(file).endswith('.png'):
		with open('list_imgs.csv', 'a') as op_file:
			op_file.write(str(file) + '\n')