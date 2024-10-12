def message ():
    """ Display a simple message"""
    print ('I am Arthur')
    print ('King of the Britons')
message ()

#Next example using a parameter
def message (king_name):
    """ Display a message with parameter"""
    print ('I am', king_name)
    print ('King of the Britons')
message ('Henry VIII')

#Next example with positional arguments
def message (king_name, kingdom):
    """ Display a message with positional arguements"""
    print ('I am', king_name)
    print ('King of the', kingdom)
message ('Mufasa', 'Lions')
message ('Elvis', 'Rock and Roll')
#Be careful of positional arguments, order matters!

#example with keyword arguments
def message (king_name, kingdom):
    """ Display a Keyword arguement"""
    print ('I am', king_name)
    print ('King of the', kingdom)
message (kingdom='Lions', king_name='Mufasa')

# Example with default value
def message (king_name, kingdom='Egypt'):
    """ Display a default value"""
    print ('I am', king_name)
    print ('King of', kingdom)
message ('Tut')

# example of default values
def message (king_name, kingdom='Egypt'):
    """ Display a default value"""
    print ('I am', king_name)
    print ('King of', kingdom)
message ('Tut')
message ('Ramesses')
message ('Thumoset')
message ('Other Guy')