pa= 80000
tca=0.03
pb=200000
tcb=0.015

ano=0

while pb>pa:
    pa = pa + pa * tca
    pb = pb + pb * tcb
    ano+=1

print(ano)