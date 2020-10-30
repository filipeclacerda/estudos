from functools import reduce
lista = range(1, 9)
pares = list(filter(lambda x: x % 2 == 0, lista))
maior = lambda x, y: x if x>y else y
print("O maior número par é:", reduce(maior, lista))


