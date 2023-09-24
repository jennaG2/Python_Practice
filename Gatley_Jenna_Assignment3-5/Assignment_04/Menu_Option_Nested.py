Menu_option = ''

while Menu_option!= 'q':
    print('''
    Shop information FAQS:
    a: cut and fold brochures
    b:deliver print jobs 
    q:quit''')
    Menu_option= input("Enter a letter for more info around the shop. ")
    if Menu_option== 'a':
        print('The cutter and folder can be dangerous. Get training before using')
    elif Menu_option== 'b':
        van_driver= input('Are you comfortable driving a class B van? Enter (y or n): ')
        if van_driver== 'y':
            print("Awesome! It would be great to have you help deliver on occasion!")    
        else:
            print('We wont ask you to drive!')   