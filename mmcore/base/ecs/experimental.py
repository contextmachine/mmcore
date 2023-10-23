from itertools import count

COUNTER = count()
entities = []


class Entity:
    def __init__(self):
        self.uid = next(COUNTER)
        entities.append(self)


Entity(), Entity(), Entity()
print(entities)
