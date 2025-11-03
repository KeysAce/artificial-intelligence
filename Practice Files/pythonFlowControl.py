var1 = 12
var2 = 8

if var1 > var2:
    print(f"{var1} is greater than {var2}")
elif var1 < var2:
    print(f"{var2} is greater than {var1}")
else:
    print(f"{var1} and {var2} are equal")

lang = input("Your programming language: ")

match (lang.lower()):
    case "python":
        print("AI")
    case "c++":
        print("Codelab")

for i in range(1,6):
    print(i)

i = 0
while i < 5:
    print(i)
    i += 1

def greet():
    print("Welcome to the program")

def addNums (x,y):
    print(f"{x} + {y} = {x+y}")

greet()
addNums(3,5)