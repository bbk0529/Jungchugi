import numpy as np
fr = open('jung.txt', encoding='utf-8')
problem=[]
arrLine=[]
for line in fr.readlines() :
	line=line.replace('\ufeff','')
	line=line.replace('\n','')
	arrLine=line.split(sep=':')
	problem.append(arrLine)

while True:
	n=np.random.randint(0,len(problem)-1)
	print('================================================')
	print(problem[n][1])
	a=input()
	if a=='q' : break
	print(problem[n][0] + '\n\n')
