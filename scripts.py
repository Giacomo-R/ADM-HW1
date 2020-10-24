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
 


