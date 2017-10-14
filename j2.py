import pickle
import glob

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


def read_wrong(filename) : #파일에서 번호를 읽어와서 배열로 만듬. 
	f=open(filename,'rb')
	arr_wrong= pickle.load(f)
	f.close()
	return arr_wrong
	
def write_wrong(filename, arr_wrong) :
	f=open(filename, 'wb')
	pickle.dump(arr_wrong, f)
	f.close()


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
		o= read_wrong('r_' + filename)
		print (o)

	fname2='r_' + filename 
	return [o, a, p, fname2]

	
def q_and_a (inp):
	again=[]
		
	print ('\n','='*50, '\n', '\n', '='*50,'\n')
	for pick in inp[0]:
		print(inp[2][pick]);input()
		print(inp[1][pick]);temp=input()
		if temp=='q': break
		if temp=='a': 
			again.append(pick)
	if len(again) !=0 : 
		if len(glob.glob(inp[3]))!=0: 
			again.extend(read_wrong(inp[3]))
		write_wrong(inp[3],again)
	
def start1():
	q_and_a(read_problem('test.txt'))
	
	
def start1_r():
	q_and_a(read_problem('test.txt', True))
	