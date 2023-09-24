a = 4 # a can be any digit > 1 and < 10
b = 6  # b can be any other digit > 2 and < 10 
c = 2

# a can be any digit > 1 and < 11 
# b can be any other digit > 2 and < 10 
# c must be 2

''' Multi-line comment example
This program will throw an error if the variables are not set first. 
After you set these variables the complete and incomplete 
print statements will run to give you a head start.
'''

# The following is a list of all the math operations. 

# Shown is Addition and a tab then show a '+' b '=' a + b. 
# Notice how the operators must be in quotes to be treated as plain text
# and not operators.
print('Addition:\t', a, '+', b, '=', a + b)



# Subtraction
print('Subtraction:\t', a, '-', b, '=', a - b)

# Multiplication
print('Multiplication:\t', a, 'x', b, '=', a * b)

# Division 
print('Division:\t', a, '÷', b, '=', a / b)

# Floor Division 
print('Floor Division:\t', a, '÷', b, '=', a // b)

# Display remainder after division (Modulo)
# May need extra \t because Modulo is the shortest label.
print('Modulo:\t\t', a, '%', b, '=', a % b)

# Display Exponent a + b squared. Make it fit the output pattern. 
# Use variable c in the code, not an actual number 2.

# u'\xb2' is the python escape for 2 superscript. Use this to 
# output the 2 superscript.

# Set it as a string literal variable.

# Call the string literal variable in the print function.



Exponent_Symbol = u'\xb2'
#The following comment is my original Exponent attempt where I incorrectly used f string
#Print(f'Exponent:\t\t', a, '+', b,{Exponent_Symbol}, '=', a + b**c)

Exponent_Symbol = u'\xb2'
print(f'Exponent:\t\t {a} + {b} {Exponent_Symbol} = {a + b**c}')


'''
Division with currency output with f string: {a/b:???} = $ ?.??
Don't forget to use labels for consistent output. Add a new line at
the start of this string to separate it from the previous strings.
Sample:

Division 
(dollars):       10 ÷ 3 = $3.33
'''

Total_Price = a / b
print(f'Division (dollars):\t {a} ÷ {b} = ${Total_Price:.2f}')