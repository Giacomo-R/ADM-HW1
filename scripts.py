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
 

thickness = int(input()) #This must be an odd number
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




#




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
    # your code goes here
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
    # your code goes here
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
    # your code goes here
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


#



#




#



#



#



#




#



#




#



#
 
    
    


