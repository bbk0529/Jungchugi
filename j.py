def cls():
	print("\n" * 100)



def start(filename='test.txt', descriptive=False) : 
	#cls()
	import numpy as np
	fr = open(filename, encoding='utf-8')
	tempLine=''
	problem={}
	answer=''
	p={}

	for currLine in fr.readlines() :
		if currLine.find('[')!=-1 : #Answer line
			if answer!='' : 
				problem[answer] = tempLine
				tempLine=''
			answer=currLine
				
		else : 
			tempLine +=currLine
	
	
	
	while True:
		pick=np.random.choice(list(problem.keys()))
		
		if descriptive==False: 
			print(problem[pick])
			a=input()
			print(pick)
			a=input()
			#cls()
		else :
			print(pick)
			a=input()
			print(problem[pick])
			a=input()
			#cls()
			
		if a=='q': break


def convert(fname):
	f=open('new%s' %fname,'w')
	for x in open(fname).readlines() :
	    if x.find('.')<4 : 
	        f.write(x.replace('.','['))
	    else : f.write(x)



def start1() : start('test.txt')
def start2() : start('short_answer.txt')
def start3() : start('desc_answer.txt', True)
def start4() : start('newtest2.txt')
