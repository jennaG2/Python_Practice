while True:   

    file_variable = input('''
                    What Would you like to search? 
                       A- 80s Music 
                       B- Athletes 
                       X-Exit. ''' )
    
    if file_variable == 'x':
       break
    elif file_variable == 'a':
       file_variable = '80s-music.txt'
    elif file_variable == 'b':
        file_variable = 'athletes.txt'
    else:
        print("Invalid option. Please select a, b, or x.")
        continue

    search_variable = input(f'''
                            Enter the search term for {file_variable} file: ''')
    def search_str(file_variable, search_variable):
        with open(file_variable, 'r') as file:
            content = file.read()
            if search_variable in content:
                print(f'Your search term {search_variable} exists in the {file_variable} file!')
                Y = input(f'\nWould you like to see the ntries? (y or n): ')
                if Y.lower() == 'y':
                    print(f'\nHere are all the entries associated with the term {search_variable}:')
                    with open(file_variable, 'r') as file:
                        for line in file:
                            if search_variable in line:
                                print(line.strip())
                elif Y.lower() == 'n':
                    print('Bye!')
                else:
                    print(f'{search_variable} does not exist in {file_variable}')
    search_str(file_variable, search_variable)