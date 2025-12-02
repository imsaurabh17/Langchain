from typing import TypedDict

class Person(TypedDict):

    name : str
    profession : str

p1: Person = {"name": "Saurabh","profession": "AI Engineer"}

print(p1)