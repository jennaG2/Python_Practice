name = "Jenna"
page_topic = "Interests"

interests = ['Swimming', 'Hiking', 'Painting']
for interest in interests:
    print(interest)

print(f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} | {page_topic}</title>
</head>
<body><hi>{page_topic}</hi>
''')

from interests_module import interests_loop

print(f'''
</body>
</html>
      ''')