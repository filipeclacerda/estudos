import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
df = pd.read_csv('esic.csv', engine="python", index_col=False, header=0, delimiter=";", encoding='utf-8-sig')

#print(df.dtypes)
#print(df.shape)
print(df.columns)
#print(df.index)

#print(df.tail(n=4))
#print(df[['sexo', 'nomecurso', 'municipio', 'bairro']])
#print(df[df["Situacao"]=="EVADIDO"][["IRA", "municipio", "nomecurso"]])
#print(df['Situacao'].value_counts())

#EMPTY CELLS TRATADAS
df.dropna(subset=['IRA', 'sexo'], inplace=True)

print(df.info)
print(df.head())

print(df.sort_values(by=['IRA']))

print(df.nlargest(5, ['IRA']))


contCidade = df["municipio"].value_counts()


fig1 = px.pie(df, names="sexo", title="sexo")
fig2 = px.pie(df, names="anoing", title="ano de ingresso")
fig3 = px.pie(df, names="Situacao", title="situação")
fig4 = px.bar(x=contCidade.index,y=contCidade.values, title="Municipios", labels=dict(x ="Municipio", y="Quantidade de alunos"))
fig4.update_layout(xaxis=dict(range=[0, 35]))

#fig1.show()
#fig2.show()
#fig3.show()
#fig4.show()

print(df.filter(items=['sexo', 'grandearea']))


fCurso = df[df['sexo']=='F'].groupby("grandearea").size()
print(fCurso)

mCurso = df[df['sexo']=='M'].groupby("grandearea").size()
print(mCurso)

sec = pd.concat((fCurso, mCurso), axis=1)
sec.fillna(0, inplace=True)
print(df.columns)
sec.columns = ['F', 'M']
print(sec)
df['Quantidade'] = 1
fig = px.bar(df, x="grandearea", y='Quantidade',
             color='sexo', barmode='group',
             height=400)
#fig.show()
sec1 = sec.drop(['GRADUAÇÃO PRESENCIAL'])

fig10 = go.Figure(
    data=[
        go.Bar(
            name="Feminino",
            x=sec1.index,
            y=sec1["F"],
            offsetgroup=0,
        ),
        go.Bar(
            name="Masculino",
            x=sec1.index,
            y=sec1["M"],
            offsetgroup=1,
        ),
    ],
    layout=go.Layout(
        title="Distribuição de Alunos entre as Áreas",
        yaxis_title="Numero de Alunos"
    )
)
fig10.show()
#sec.rename(columns={'count':'Total_Numbers'})
#contaFeminino = df['sexo'].value_counts()['F']
#contaMasculino = df['sexo'].value_counts()['M']


#print(contaFeminino)