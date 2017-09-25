import numpy as np
fr = open('test.txt')
#fr = open('test.txt', encoding='utf-8')
problem=[]
tempLine=''

for currLine in fr.readlines() :
	if currLine.find('*')!=-1 :
		if tempLine !='' : 
			problem.append(tempLine.split(':'))
		tempLine = currLine
	else : 
		tempLine +=currLine

while True:
	n=np.random.randint(0,len(problem)-1)
	print('================================================')
	print(problem[n][1])
	a=input()
	if a=='q' : break
	print(problem[n][0] + '\n\n')
