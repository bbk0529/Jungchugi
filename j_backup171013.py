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
	

def read_problem(filename='test.txt', onlywrong=False) : #onlyWrong = 틀린문제만 읽어들임. 
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
	
	a=list(problem.keys())
	p=list(problem.values())
	
	if onlywrong == False : 
		o=np.random.choice(np.arange(0,len(p)), len(p), False)
	else : 
		fname2='r_' + filename 
		o= again(fname2)
		print(o)
	return o, a, p, filename

	
def q_and_a (o, a, p, filename, repeat, descriptive=False, recording=False):
	again=[]
	
	if recording ==True :
		print(filename)
		f=open(filename,'a')
		f.write('\n')
		
	print ('\n','='*50, '\n', 'Repeated number  ', repeat, '\n', '='*50,'\n')
	for pick in o:
		if descriptive==False: 
			print('\n'*5, repeat, ' ', pick, '\n', p[pick]);	input()
			print(a[pick]); 							temp=input()
		else :
			print('\n'*5, repeat, ' ',pick,'\n', a[pick]);	input()
			print(p[pick]); 							temp=input()
			
			
		if temp=='q': break
		if temp=='a': 
			again.append(pick)
			if recording == True : 
				pick= str(pick) + '\n'
				f.write(pick)
		if len(again) !=0 : 
			q_and_a(again, a, p, filename, repeat+1)
	f.close()		
	

def start1():
	o,a,p,filename=read_problem('test.txt')
	q_and_a(o,a,p,filename,1)
	


def start2():
	o,a,p,filename=read_problem('short_answer.txt')
	q_and_a(o,a,p,filename,1)


def start3():
	o,a,p,filename=read_problem('desc_answer.txt', True)
	q_and_a(o,a,p,filename,1)
	