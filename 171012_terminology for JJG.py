def cls():
	print("\n" * 100)

def start(filename) : 

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
		p[pick]=p.get(pick,0) +1		
		print("="*80 + "\n") 
		print(problem[pick])
		a=input()
		print(pick)
		a=input()
		cls()
		if a=='q' :  return p