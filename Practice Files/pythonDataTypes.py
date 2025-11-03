name = "Keolan"
age = 19
height = 180.33
studentStatus = True

if studentStatus == True:
    print("My name is", name + ". I am", str(age), "years old,", str(height) + "cm tall and I am a student")
else:
    print("My name is", name + ". I am", str(age), "years old,", str(height) + "cm tall and I am not a student")

age += 5
height /= 2

print("New age:",str(age))
print("New height:",str(height))
