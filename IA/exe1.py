#calcule a soma dos numeros pares entre 120 e 150

soma = 0
for item in range(102, 151, 2):
    soma = soma + item
    print(item)
print("a soma é " + str(soma))