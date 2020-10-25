# Say "Hello, World!" With Python

if __name__ == '__main__':
    print ("Hello, World!")
    
# Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())

if n%2 : 
    print ("Weird")
    exit()
if n%2 == 0 : 
    if n in range(2,6):
        print ("Not Weird")
        exit()
    if n in range (6,21):
        print ("Weird")
        exit()
    if n > 20 :
        print ("Not Weird")
        exit()


# Arithmetic Operators

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())

print (a+b)
print (a-b)
print (a*b)


# Python: Division

from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
print (a//b)
print (a/b)


# Loops


if __name__ == '__main__':
    n = int(raw_input())

    i = 0 
    while i < n : 
        print (i**2)
        i = i+1 



# Write a function

def is_leap(year):
    leap = False
    if not year%4 : 
        leap = True  
        if not year%100 : 
            leap = False
            if not year%400 :
                leap = True 
    
    return leap



# Print Function

from __future__ import print_function

if __name__ == '__main__':
    n = int(raw_input())

i = 1
j = str(i)
while i < n : 
    i = i+1
    j = j + str(i)

print(j)    


# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

L = [[v,w,m] for v in range(0,x+1) for w in range(0,y+1) for m in range(0,z+1)]  

W = [x for x in L if sum(x) != n]
print(W)



# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())

t = list(set(arr))
t.remove(max(t))
print(max(t))




# Nested Lists


K = []
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        K.append([score,name])

K.sort()

for i in range (1,len(K)):
    if K[0][0] == K[i][0]:
        K.pop(i)
    else : 
        x = K[i][0]
        break 

G =[]
for y in K: 
    if y[0] == x: 
        G.append(y[1])

G.sort()
for i in range (0,len(G)):
    print (G[i])




# Finding the percentage

if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()


print('%.2f' % (sum(student_marks[query_name])/3))




# Lists

if __name__ == '__main__':
    N = int(input())
    L =[]
for i in range(0,N): 
    attr = list(input().split())
    if attr[0] == 'print':
        print (L)
    else:
        if len(attr) == 1: 
            getattr(L, attr[0])()
        if len(attr) == 2:
            getattr(L,attr[0])(int(attr[1]))
        if len(attr) == 3:
            getattr(L, attr[0])(int(attr[1]), int(attr[2]))


# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = tuple(map(int, input().split()))
    print (hash(integer_list))




# sWAP cASE

def swap_case(s):
    i = 0
    
    while(i<len(s)) : 
        if  ord(s[i]) in range(97,123)  : 
            s = s[:i] + chr (ord(s[i]) - 32) + s[i+1: ]
            
        elif  ord(s[i]) in range(65,91)  : 
            s = s[:i] + chr (ord(s[i]) + 32) + s[i+1: ]       
        
        i = i+1   
    return s 




# String Split and Join

def split_and_join(line):
    # write your code here
    line = line.split(" ")
    line = "-".join(line)

    return line 



# What's Your Name?

def print_full_name(a, b):
   print("Hello %s %s! You just delved into python." % (a,b))




# Mutations

def mutate_string(string, position, character):
    
    string = string[:position] + character + string[position+1: ]
    
    return string



# Find a string

def count_substring(string, sub_string):
    i_old = -1
    cont = 0
    
    while True :  
        i_new = string.find(sub_string, i_old+1)
        if (i_new > i_old) : 
            cont = cont + 1
            i_old = i_new  
        else : break 
    return cont 

# String Validators

if __name__ == '__main__':
    s = input()


i = 0

while True :
    if s[i].isalnum() :
        print ("True")
        break 
    else :
        i = i+1
    if i >= len(s) :
        print("False")  
        break

i = 0

while True :
    if s[i].isalpha() :
        print ("True")
        break 
    else :
        i = i+1
    if i >= len(s) :
        print("False")    
        break 


i = 0

while True :
    if s[i].isdigit() :
        print ("True")
        break 
    else :
        i = i+1
    if i >= len(s) :
        print("False")    
        break


i = 0        

while True :
    if s[i].islower() :
        print ("True")
        break 
    else :
        i = i+1
    if i >= len(s) :
        print("False")    
        break

i = 0

while True :
    if s[i].isupper() :
        print ("True")
        break 
    else :
        i = i+1
    if i >= len(s) :
        print("False")    
        break




# Text Alignment
 

thickness = int(input()) 
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))


for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    


for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))




# Text Wrap

def wrap(string, max_width):
    n = len(string) // max_width
    x = 0 
    cont = 0

    while (cont < n) :
        y = x + max_width
        print (string[x:y])
        cont = cont + 1 
        x = x + max_width 

    return string[max_width*cont:]



# Designer Door Mat

def central_pattern (n):
    for i in range (0,2*n+1):
        print ('.|.', end='')
    return 

def lateral_pattern (n): 
    for i in range(0,n):
        print ('-', end='')
    return 

L = list(map(int, input().split()))
j = 0

while j <= L[0]//2 - 1 :
    lateral_pattern ((L[1]//2-1) - 3*j)
    central_pattern(j)
    lateral_pattern ((L[1]//2-1) - 3*j)
    print()
    j = j+1

lateral_pattern ((L[1]-7)//2) 
print ("WELCOME", end='')
lateral_pattern ((L[1]-7)//2) 
print()
j = j-1

while j >= 0 :
    lateral_pattern ((L[1]//2-1) - 3*j)
    central_pattern(j)
    lateral_pattern ((L[1]//2-1) - 3*j)
    print()
    j = j-1








# Alphabet Rangoli

def lateral_pattern(size, j):
        for i in range(0, 2*size-2*j):
            print('-', end='')

def central_pattern(size, j):
    if j == 1: print (chr(97+size-1), end='')
    else :
        for i in range(0,j):
            print (chr(97+size-(1+i)), end='')
            print('-', end='')
        for i in range(j-2,-1,-1):
            if i == 0 : 
                print (chr(97+size-(1+i)), end='')
            else :
                print (chr(97+size-(1+i)), end='')
                print('-', end='')
        
def print_rangoli(size):
   
    for j in range(1,size):
        lateral_pattern(size, j)
        central_pattern (size, j)
        lateral_pattern(size,j)
        print()
    central_pattern(size,size)
    print()
    for j in range(size-1,0,-1):
        lateral_pattern(size, j)
        central_pattern (size, j)
        lateral_pattern(size,j)
        print()


# Capitalize!

def solve(s):
    L = s.split(" ")
    d = L[0].capitalize()
    i = 1
    while i<len(L) : 
        d = d + ' ' 
        d = d + L[i].capitalize()
        i = i + 1 
    
    return d



# Merge the Tools!

from collections import OrderedDict

def merge_the_tools(string, k):
    
    L = []
    i = 0 
    x = k  
    while k <= len(string):
        L.append(string[i:k])
        i = k 
        k = k+x

    for i in range(0, len(L)):
        A = OrderedDict.fromkeys(L[i])
        st = ''.join(A.keys())
        print(st)



# collections.Counter()

from collections import Counter

n = int(input())
L = list(map(int, input().split()))
C = Counter(L)
x = 0 

for i in range(0,int(input())): 
    L = list(map(int,input().split()))
    if L[0] in C and C[L[0]] > 0 :
        C[L[0]] = C[L[0]]-1
        x = x + L[1]
    else : 
        continue 
print (x)




# Introduction to Sets

def average(array):
   
    S = list(set(array))
    Result = sum(S) / len(S)
    
    return Result




# DefaultDict Tutorial

from collections import defaultdict

L = list(map(int, input().split()))
a = defaultdict(list)
Q = []

for i in range(1,L[0]+1):
   a[input()].append(str(i))

for i in range(1,L[1]+1):
    Q.append(input())

for x in Q:
    if x in a.keys(): 
        print(*a[x])
    else : 
        print(-1)
    





# Calendar Module

import calendar 

L = list(map(int,input().split()))
T = ('MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY')
print (T[calendar.weekday(L[2],L[0],L[1])])



# Exceptions


n = int(input())

for i in range(0,n):
    try:
        L = list(map(int,input().split()))
        print (L[0]//L[1])
    except ZeroDivisionError as e:
        print ("Error Code:",e)
    except ValueError as e:
        print ("Error Code:",e) 





# Collections.namedtuple()

from collections import namedtuple
n = int(input())
L = input().split()
table = namedtuple('table', L)
x = 0 

for i in range (0,n):
    student = table(*input().split())
    x = x + int(student.MARKS)

print ('%.2f' % (x / n) )





# No Idea!

n, m = input().split()

sc_ar = input().split()

A = set(input().split())
B = set(input().split())
print (sum([(i in A) - (i in B) for i in sc_ar]))



# Collections.OrderedDict()

from collections import OrderedDict

n = int(input())
d = OrderedDict()

for i in range(0,n):
    L = input().split()
    word = " ".join(word for word in L[:-1])

    if word in d: 
        d[word] = int(L[-1]) + int(d[word])
    else :
        d [word] = int(L[-1])

for x in d.keys():
    print ("%s %s" %(x, d[x]) )




# Symmetric Difference


n, A = input(), set(input().split())

m, B = input(), set(input().split())



L1 = [int(x) for x in list(A.difference(B))]
L2 = [int(x) for x in list(B.difference(A))]

L1 = L1 + L2 
L1.sort()

i = 0 
while i<len(L1) : 
    print (L1[i])
    i = i+1






# Set .add()


n = int(input())

s = set()

i = 0 

while i < n : 
    s.add(input())
    i = i+1

print (len(list(s)))     





# Word Order

from collections import OrderedDict 

n = int(input())
d = OrderedDict()

for i in range(0,n):
    word = input()
    if word in d : 
        d[word] = d[word] + 1 
    else :
        d [word] = 1 

print (len(d.keys()))
print (*d.values())





# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))

c = int(input())
i = 0

while i < c : 
    a = input()

    if a[0] == 'p' : 
        s.pop()   
    if a[0] == 'r' :
        s.remove(int(a[7]))
    if a[0] == 'd' :
        s.discard(int(a[8]))
    
    i = i+1 

print(sum(list(s)))



# Collections.deque()

from collections import deque
n = int (input())
d = deque()

for i in range(0,n):
    L = input().split()
    if len(L) > 1 : getattr (d, L[0]) (int(L[1]))
    else : getattr (d, L[0]) ()
    
print (*d)





# Company Logo


import math
import os
import random
import re
import sys
from collections import Counter
import operator

if __name__ == '__main__':
    s = input()
C = Counter(s) 
L = []

for key,value in C.items():
    l = [key, int(value)]
    L.append(l)

L.sort (key=operator.itemgetter(0))
L.sort(key = operator.itemgetter(1), reverse=True)

for x in L[:3]: 
    print(*x)




# Set .union() Operation


_,a = input(), set(map(int,input().split()))
_,b = input(), set(map(int,input().split()))

print(len(list(a|b)))


# Piling Up!

N = int(input())

for i in range(0,N): 
    _ = input()
    L = list(map(int,input().split()))
    x = L.index(min(L))
 
    L1 = sorted(L[x:])
    L2 = sorted(L[:x], reverse=True)
    
    if L1 == L[x:] and L2 == L[:x] : 
        print ('Yes')
    else :
        print('No')


# Set .intersection() Operation



_,a = input(), set(map(int,input().split()))
_,b = input(), set(map(int,input().split()))

print(len(list(a&b)))



# Set .difference() Operation



_,a = input(), set(map(int,input().split()))
_,b = input(), set(map(int,input().split()))

print(len(list(a-b)))



# Set .symmetric_difference() Operation


_,a = input(), set(map(int,input().split()))
_,b = input(), set(map(int,input().split()))

print(len(list(a^b)))



# Set Mutations

_ = input()
A = set(map(int, input().split()))
N = int(input())

for i in range(0,N):
    attr = input().split()
    H = set(map(int, input().split()))
    getattr(A, attr[0])(H)

print(sum(list(A)))


# The Captain's Room



from collections import Counter

_, L = input(), list(map(int, input().split()))
A = Counter(L)

for x,y in A.items():
    if y == 1:
        print(x)



# Check Subset


n = int(input())
L = [0]*(n)
M = [0]* (n)
i = 0
while i < n : 

    _, L[i] = input(), set(map(int, input().split()))
    _, M[i] = input(), set(map(int, input().split()))
    print (L[i] < M[i])
    i = i + 1 


# Check Strict Superset


A = set(map(int, input().split()))
n = int(input())
i = 0 

while i < n : 
    B = set(map(int, input().split()))
    if not(B < A) : 
        print("False")
        exit()
    i = i + 1 

print ("True")




# Zipped!


L = list(map(int,input().split()))
K = []

for i in range(0,L[1]):
    G = list(map(float,input().split()))
    K.append(G)

for x in zip(*K):
    print (sum(x)/len(x))



# Athlete Sort
 
import math
import os
import random
import re
import sys
import operator


if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

arr.sort(key=operator.itemgetter(k))

for x in arr:
    print(*x)


# ginortS


txt = input()
L = [] 

for i in range(0,len(txt)):
    if txt[i].islower():
       a = '0'+txt[i]
       L.append(a)
    
    if txt[i].isupper():
        a = '1' + txt[i]
        L.append(a)
    
    if txt[i].isdigit():
        if int(txt[i])%2:
            a = '2' + txt[i]
            L.append(a)
        else : 
            a = '3' +txt[i]
            L.append(a)

L.sort()

for x in L: 
    print (x[1:], end='')


    

    
    

# Detect Floating Point Number

import re 

n = int(input())

for i in range(0,n):
    txt = input()
    m = re.match('^([+\-.]{0,1})\d{0,}\.\d+$' , txt)
    if m:
        print('True')
    else:
        print('False')





# Map and Lambda Function   
    

def fibonacci(n):
    
    L = [0,1]
    if n == 0: 
        return []
    if n == 1 : 
        return [0]
    if n == 2 :
        return L
    i = 2 
    while i < n :
        L.append(L[i-1] + L[i-2])
        i = i +1 
    return L          



    
    

# Re.split()


regex_pattern = r"[. ,]"

import re
print("\n".join(re.split(regex_pattern, input())))



# Re.findall() & Re.finditer()


import re

c_list = re.findall('(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])[AEIOUaeiou]{2,99}(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])', input())


if not c_list : 
    print (-1)
else: 
    for x in c_list:
       print(x)

    
    

    
    

# Re.start() & Re.end()

import re 
txt1= input()
a = input()
txt2 = re.compile(a)
j = 0

if not re.search(txt2, txt1) : print ((-1,-1))

if len(a) == 1 : 
    while True : 
        m = re.search(txt2, txt1[j:])
        if not m : break 
        print((m.start()+j, m.end()-1+j))
        j = m.end()+1+j  

else :
    while True : 
        m = re.search(txt2, txt1[j:])
        if not m : break 
        print((m.start()+j, m.end()-1+j))
        j = m.end()-1+j 






# Validating phone numbers 


import re 

n = int(input())

for i in range(0,n):
    txt = input()
    h = len(txt)
    if not h == 10: 
        print('NO')
        continue 
    if re.match('[7-9]', txt) : 
        if re.search('[0-9]{9}', txt[1:]):
            print('YES')
        else :
            print('NO')
    else: 
        print('NO')


    
    

# Validating and Parsing Email Addresses


import re 
n = int (input())
for i in range(0,n):     
    txt = list(input().split()) 
    m = re.match ('^([A-Za-z])([a-zA-Z.\-_0-9]+)@([a-zA-Z]+)\.([a-zA-Z]{1,3})$', txt[1][1:-1])
    if m :
        print (*txt)



# HTML Parser - Part 1

  
    

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self, tag, attrs):
        print ("Start :", tag)
        for i in range(0, len(attrs)):
            print ('->', attrs[i][0], end='')
            print(' >', attrs[i][1])

    def handle_endtag(self, tag):
        print ("End   :", tag)
    
    def handle_startendtag(self, tag, attrs):
        print ("Empty :", tag)
        for i in range(0, len(attrs)):
            print ('->', attrs[i][0], end='')
            print(' >', attrs[i][1])


n = int(input())
parser = MyHTMLParser()
txt = ''

for i in range(0,n):
    txt = txt + input()
    
parser.feed(txt)

    
    

# HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_comment(self, data):
        L = data.split('\n')
        if len(L) > 1: 
            print('>>> Multi-line Comment')
            for i in range(0,len(L)):
                print(L[i])
        else: 
            print('>>> Single-line Comment')
            print(data)

    def handle_data(self, data):
        if  data != '\n' : 
            print('>>> Data')
            print(data)

  
  
  
  
  
  
  
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()



#   Detect HTML Tags, Attributes and Attribute Values


from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    
    def handle_starttag(self, tag, attrs):
        print ('', tag, sep='')
        for i in range(0, len(attrs)):
            print ('->', attrs[i][0], end='')
            print(' >', attrs[i][1])

    def handle_endtag(self, tag):
        x = tag
    
    def handle_startendtag(self, tag, attrs):
        print ("", tag, sep='')
        for i in range(0, len(attrs)):
            print ('->', attrs[i][0], end='')
            print(' >', attrs[i][1])


n = int(input())
parser = MyHTMLParser()
txt = ''

for i in range(0,n):
    txt = txt + input()
    
parser.feed(txt)

    


    
    

# XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
   
    x = len(node.attrib)
    n = count_attr (node, [x])
    return n 

def count_attr (node, L):

    if node == None : return 0

    for child in node: 
        count_attr(child, L)
        L.append(len(child.attrib))

    return (sum(L)) 
    
    

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))





#   XML2 - Find the Maximum Depth
    

import xml.etree.ElementTree as etree
    
    
maxdepth = 0
def depth(elem, level):
    global maxdepth
    
    x = count_levels (elem, [0])
    maxdepth = x 

def count_levels (node, L): 

    if list(node) == [] :
        return 0 
    else : 
       L[0] = L[0] + 1 


    for child in node: 
       count_levels(child, L)
        
    
    L.insert(0,1)
    
    return(max(L))
    
    
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
    
    

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        
        g = []
        for x in l:
            x = x[len(x)-10:]
            x = '+91' + ' ' + x[0:5] + ' ' + x[5: ]
            g.append(x)
        f(g)
        

    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 



#   Decorators 2 - Name Directory
import operator 

def person_lister(f):
    def inner(people):
        for i in range(0,len(people)):
            people[i][2] = int(people[i][2])
        L = sorted(people, key=operator.itemgetter(2))
        return map(f, L)
    return inner


    
@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')
    
    

# Arrays
import numpy

def arrays(arr):
   
    arr.reverse()
    a = numpy.array(arr,float) 
    return a 


arr = input().strip().split(' ')
result = arrays(arr)
print(result)



# Shape and Reshape
import numpy  


L = list(map(int,input().split()))
arr = numpy.array(L)
print (numpy.reshape(arr,(3,3)))

    
    

# Transpose and Flatten

import numpy

P = list(map(int,input().split()))
Q = []


for i in range(0,P[0]):
    L = list(map(int,input().split()))
    Q.append(L)

arr = numpy.array(Q)

print (numpy.transpose(Q))
print (arr.flatten())



# Concatenate

import numpy

L = list(map(int,input().split()))
Q = []

for i in range(0,L[0]+L[1]):
    A = list(map(int,input().split()))
    Q. append(A)

arr = numpy.array(Q)
print(arr.reshape(L[0]+L[1], L[2]))

    

    
    

# Zeros and Ones



import numpy


T = tuple(map(int,input().split()))

print(numpy.zeros(T, dtype = int))
print(numpy.ones(T, dtype = int))


# Eye and Identity
  
    
# I added line 8 from the suggestion by @nprajilesh (in the "Discussion" section) 
# the rest of the program is my own but w/o this suggestion i wouldn't be able to match 
# my output w/ the output given by HackerRank

import numpy

L = list(map(int, input().split()))
numpy.set_printoptions(sign=' ')
print (numpy.eye(L[0], L[1], 0))
    
    

# Array Mathematics
import numpy

L = list(map(int, input().split()))
Q = []

for i in range(0,L[0]):
    P = list(map(int,input().split()))
    Q.append(P)
arr1 = numpy.array(Q)

K = []
for i in range(0,L[0]):
    P = list(map(int,input().split()))
    K.append(P)
arr2 = numpy.array(K)

print (arr1 + arr2)
print (arr1 - arr2)
print (arr1 * arr2)
print (arr1 // arr2)
print (arr1 % arr2)
print (arr1 ** arr2)




#   Floor, Ceil and Rint
    
# On line 8 i reused the idea of @nprajilesh that i used in the "Eye and Identity" exercise from the NumPy section  



import numpy

arr = numpy.array(list(map(float,input().split())))
numpy.set_printoptions(sign=' ')  
print (numpy.floor(arr))
print (numpy.ceil(arr))
print (numpy.rint(arr))
    
    

# Sum and Prod

import numpy


P = list(map(int,input().split()))
Q = []
for i in range(0,P[0]):
    L = list(map(int,input().split()))
    Q.append(L)

arr = numpy.array(Q)

print (numpy.prod(numpy.sum(arr, 0), None)) 



#   Min and Max
    
import numpy

P = list(map(int,input().split()))
Q = []
for i in range(0,P[0]):
    L = list(map(int,input().split()))
    Q.append(L)

arr = numpy.array(Q)

print (numpy.max(numpy.min(arr, 1), None))

    
    

# Mean, Var, and Std

# In line 6 i reused the suggestion used in "Eye and identity" from @nprajilesh, i also
# used the idea from @codeharrier (and others) in line 14. Before my idea was to print 
# in line 20 using: print ("%.11f" % (numpy.std(arr, None)) )
# the problem w/ this is that the output given by HackerRank first uses precision .11 
# and after uses precision .12, so my idea doesn't work  


import numpy


numpy.set_printoptions(sign=' ')
P = list(map(float,input().split()))
Q = []
for i in range(0,int(P[0])):
    L = list(map(float,input().split()))
    Q.append(L)

numpy.set_printoptions(legacy='1.13')
arr = numpy.array(Q)

print (numpy.mean(arr,1)) 
print (numpy.var(arr,0))    
print (numpy.std(arr, None))





# Dot and Cross  
    
import numpy
P = int(input())

Q = []
for i in range(0,P):
    L = list(map(int,input().split()))
    Q.append(L)
arr1 = numpy.array(Q)

K = []
for i in range(0,P):
    L = list(map(int,input().split()))
    K.append(L)
arr2 = numpy.array(K)

print (numpy.dot(arr1,arr2))


    
    

# Inner and Outer

import numpy

A = numpy.array(list(map(int, input().split())))
B = numpy.array(list(map(int, input().split())))

print (numpy.inner(A,B))
print(numpy.outer(A,B))





# Polynomials

  
    
import numpy

A = list(map(float, input().split()))
n = int(input()) 

print (numpy.polyval(A, n))


# Linear Algebra

# In line 8 we used the same idea used in "Mean, Var, and Std" from @codeharrier 
# and others

import numpy


P = int(input())
numpy.set_printoptions(legacy='1.13')

Q = []
for i in range(0,P):
    L = list(map(float,input().split()))
    Q.append(L)

arr = numpy.array(Q)

print (numpy.linalg.det(arr)) 


# Birthday Cake Candles


import math
import os
import random
import re
import sys
from collections import Counter

def birthdayCakeCandles(candles):
    C = Counter(candles)
    m = max(C.keys())
    return C[m]



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)
    print (result)

    fptr.write(str(result) + '\n')

    fptr.close()



# Number Line Jumps

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):

    if (v1 - v2) == 0 and (x1 - x2) == 0:
        return 'YES'
    elif v1 - v2 == 0 and x1 != x2: 
        return 'NO'

    x = (x2 - x1) / (v1 - v2)

    if x.is_integer() and x >= 0:  
        return 'YES'
    else: 
        return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()



# Viral Advertising


import math
import os
import random
import re
import sys

def viralAdvertising(n):
    x = 5 // 2
    sum = x  
    for i in range(0,n-1):
        x = 3*x // 2
        sum = sum + x 
    return sum 


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()




# Insertion Sort - Part 1


import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    m = arr[n-1]
    j = n-1
    for i in range(n-2, -1, -1):
        if m < arr[i]:
            arr[j] = arr[i]
            j = j-1
            print(*arr)
        else : 
            break
    arr[j] = m 
    print(*arr) 
         



if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)





# Insertion Sort - Part 2


import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1, len(arr)):
        j = i   
        while j>0:
            if arr[j-1] > arr[j]:
                x = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = x
                j = j-1
            else : 
                break
        print(*arr)


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)






# Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    L = list(n)
    if len(L) == 1: return int(L[0])
    g = sum(map(int,L))
    return superDigit (str(k*g),1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# The Minion Game
import re 

def minion_game(string):
    # your code goes here
    vocals = 'AEIOU'
    consonants = 'BCDFGHJKLMNPQRSTVWXZY'
    sumK = 0 
    sumS = 0 
    for x in vocals:
        indexes = re.finditer(x, string)
        for q in indexes: 
            sumK = sumK + len(string)-q.start()
    for x in consonants:
        indexes = re.finditer(x, string)
        for q in indexes: 
            sumS = sumS + len(string)-q.start()
    
    if sumS > sumK:
        print ('Stuart', sumS)
    if sumK > sumS:
        print('Kevin', sumK)
    if sumK == sumS:
        print('Draw')
        

if __name__ == '__main__':
    s = input()
    minion_game(s)


    
    
#Regex Substitution


import re 

n = int(input())

for i in range(0,n):
    txt= input()
    if txt.find('#') < 0:
        txt = re.sub (r'(?<!&{\s)(\s&{2}\s)(?!&{\s)', ' and ', txt)
        txt = re.sub (r'(?<!&{\s)(\s&{2}\s)(?!&{\s)', ' and ', txt)         
        txt = re.sub (r'(?<!\|{\s)(\s\|{2}\s)(?!\|{\s)', ' or ', txt)
        txt = re.sub (r'(?<!\|{\s)(\s\|{2}\s)(?!\|{\s)', ' or ', txt)  
        print(txt)
    else : 
        print(txt)
    



