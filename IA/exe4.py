numava = int(input("Insira o número de avaliações: "))

numaluno = int(input("Insira o número de alunos: "))

dicnotas={}

for x in range(numaluno):
    nomealuno= input("Insira o seu nome: ")
    notas=0
    for x in range(numava):
        nota = int(input("Insira a nota: "))
        notas=notas+nota/numava

    dicnotas[nomealuno] = round(notas, 2)

print(dicnotas)
x= sum(dicnotas.values())
mediaturma=x/numaluno
print("A média da turma é:", round(mediaturma, 2))



