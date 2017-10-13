def cls():
	print("\n" * 100)
def start99(filename='test.txt', descriptive=False) : 
	import numpy as np
	fr = open(filename, encoding='utf-8')
	tempLine=''
	problem={}
	answer=''

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
def again(filename) : #파일에서 번호를 읽어와서 배열로 만듬. 
	f=open(filename,'r')
	specific_number= [int(a) for a in f.readlines() if a!='\n']
	return specific_number
	

def read_problem(filename='test.txt', descriptive=False) :
	import numpy as np
	fr = open(filename, encoding='utf-8')
	tempLine=''
	problem={}
	answer=''

	for currLine in fr.readlines() :
		if currLine.find('[')!=-1 : 
			if answer!='' : 
				problem[answer] = tempLine
				tempLine=''
			answer=currLine
		else : 
			tempLine +=currLine
			
	if descriptive == False : 
		a=list(problem.keys())
		p=list(problem.values())
	else :
		p=list(problem.keys())
		a=list(problem.values())
	o=np.random.choice(np.arange(0,len(p)), len(p), False)
	return [o, a, p]


	
def start (repeat, inp):
	again=[]
	
	print ('\n','='*50, '\n', 'Repeated number  ', repeat, '\n', '='*50,'\n')
	
	for pick in inp[0]:
		print('\n'*5, repeat, ' ', pick, '\n', inp[2][pick]);input()
		print(inp[1][pick]);	temp=input()
		if temp.lower()=='a': 	again.append(pick)
		
	if len(again) !=0 : 
		inp[0]=again
		start(repeat+1,inp)
	

def start1():
	start(1,read_problem('test.txt'))
	
def start2():
	start(1,read_problem('newtest2.txt'))

def start3():
	start(1,read_problem('test3.txt'))

def start_short():
	start(1,read_problem('short_answer.txt'))

def start_desc():
	start(1,read_problem('desc_answer.txt', True))
