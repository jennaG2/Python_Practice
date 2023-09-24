#Minimum years of Adobe experience must be at least 3
#Minimum wordpress years of experience must be at least 1
#use float for room for user error

# This program determines whether a potential applicant
# qualifies for a position.

ADOBE_EXP = 3  # Minimum years of Adobe experience must be at least 3
WORDPRESS_EXP = 1  # Minimum wordpress years of experience must be at least 1

# Get the Applicant's Adobe Experience in years.
Adobe = float(input('Enter your years of experience in Adobe: '))

# Get the number of years of WordPress experience.
WordPress = float(input('Enter your years of experience in Word Press: '))

# Determine whether the customer qualifies.
if Adobe >= ADOBE_EXP and WordPress >= WORDPRESS_EXP:
    print('You qualify for this position.')
else:
    print(f'''We are looking for someone with at least: 
         * {ADOBE_EXP} years of experience in Adobe and 
         * {WORDPRESS_EXP} year of experience in WordPress.''')

