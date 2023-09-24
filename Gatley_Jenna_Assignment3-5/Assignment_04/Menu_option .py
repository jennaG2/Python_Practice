Menu_option = ''

while Menu_option!= 'q':
    print('Menu:','a: cut and fold brochures', 'b:deliver print jobs', 'q:quit', sep="\n")
    Menu_option= input("Entera letter for more info around the shop.")
    if Menu_option== 'a':
        print('The cutter and folder can be dangerous. Get training before using')
    elif Menu_option== 'b':
        print("We will give you pointers on how to drive the van!")    
print('Thank you for visitng our page!')   