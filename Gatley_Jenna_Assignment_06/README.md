## Assignment 6

This assignment is based on correcting the failed added ```else``` in the `80s-music.txt` demo I did in the Files and Exceptions class.

When I ran this code without the `else` it worked:

```python
# Define the prefix you want to filter by
contains_to_filter = "Jackson"

# Open the file for reading
with open('80s-music.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Check if the line starts with the specified prefix
        if line.__contains__(contains_to_filter):
            # Process or print lines that start with the prefix
            print(line.strip())  # For example, print the filtered line after stripping whitespace
```

This was a success. Here is the expected output:

```
Billie Jean,Michael Jackson
Don't Stop 'Til You Get Enough,Michael Jackson
Man in the Mirror,Michael Jackson
Thriller,Michael Jackson
```

However, when I added `else` and did a search on "Madonna" it did not work as expected:

```python
# Define the prefix you want to filter by
contains_to_filter = "Madonna"

# Open the file for reading
with open('80s-music.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Check if the line starts with the specified prefix
        if line.__contains__(contains_to_filter):
            # Process or print lines that start with the prefix
            print(line.strip())  # For example, print the filtered line after stripping whitespace
        else:
            print('That term is not in the file.')
```

It *sort of* worked, but it looped and printed all lines like this:

```
...
That term is not in the file.
That term is not in the file.
That term is not in the file.
Like a Prayer,Madonna
Like a Virgin,Madonna
Material Girl,Madonna
Papa Don't Preach,Madonna
That term is not in the file.
That term is not in the file.
That term is not in the file.
...
```
Strictly speaking, this is not a logical error. It is a semantic error. The semantic error could be corrected by editing the print statement to read:

```python
 print(f'{contains_to_filter}?  That term is not found in this line in the file.')
```

Then you would get this:

```
...
Madonna?  That term is not found in this line in the file.
Madonna?  That term is not found in this line in the file.
Like a Prayer,Madonna
Like a Virgin,Madonna
Material Girl,Madonna
Papa Don't Preach,Madonna
Madonna?  That term is not found in this line in the file.
Madonna?  That term is not found in this line in the file.
Madonna?  That term is not found in this line in the file.
...
```
Hmmm... okay... yes, is now logically and semantically correct, but not very "user friendly" and not a great experience.

## Challenge

The next morning, I spent several hours and I made it better, maybe not *great*, but good enough, I think. So that is the genesis of this Assignment 6 challenge. I want you to try to replicate what I did to make it better. And if you can make it better than better, or even great, then would look forward to seeing that! (Perhaps refactor it with functions; but that is not the assigment. Maybe that will be assignment 7. If not an assignment, I will definitely show you how to refactor it.)

* make the program accept any file name AND any search term using a function with those parameters.
* make it so that it would ONLY print the entries in the file that matched the search.
* make it so the user could choose between between `80s-music.txt` or `athletes.txt` with a menu
* make it so that it would loop again and again so the user could keep searching either file and any term.

In reality, this took an incredible investment of time for me. 

So to make this less time consuming I have created two files that have a mixture of pseudocode and real code. They are:

```
01_select_file_enter_term.py
02_search_for_term.py
```

Some of the real code is not complete. It is mixture of pseudocode, real code and incomplete real code. So even though you have a good framework, you it is still might be a substantial challenge. This may seem unfair, but imaging if I asked you to code it up all with just the design goals and expected output. However, the indentation hints are pretty solid.

Each one of these files will work independently from each other, but they will only work as designed if you put them together in one file. When you put them together, you will have to make sure that the function code is indented under the top while loop.

This will be worth 10 points, so it is a major project. We will spend time in class on October 23 to work on it in class. If needed, you can get help in the last part of class as the due date will be 9:00 pm October 30 at 11:55.
