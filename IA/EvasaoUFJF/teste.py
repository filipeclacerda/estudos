import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# trata os erros de digitacao do ano de conclusao e calcula o GAP
# GAP: Diferença entre o ano de conclusao e o ano de ingresso na ufjf
def calculaGAP(df):
    # trata erros de digitacção
    df['anoconc2g'] = df['anoconc2g'].replace('____', None)
    df['anoconc2g'] = df['anoconc2g'].replace('200O', 2000)
    # converte o anoconc2g para INT
    conversaoINT = lambda x: int(x)
    df["anoconc2g"] = df["anoconc2g"].apply(conversaoINT)
    # calcula o GAP
    df['GAP'] = df['anoing'] - df['anoconc2g']
    df["GAP"] = df["GAP"].apply(conversaoINT)
    # retira do dataframe GAP menores do que 0(negativos) e maiores do que 40
    # porque  eles provavelemente são erros de digitacao
    return df[(df['GAP'] >= 0) & (df['GAP'] < 40)]

#Trata variávies quantitativas contínuas
def trataQuantiContinuas(df):
    #valores com vírgulas são considerados strings
    #substitui valores lançados com vírgula para evitar erros nas contas
    virgula = lambda x: float(x.replace(',','.'))
    df["totalpontos"] = df["totalpontos"].apply(virgula)
    #após a troca, faz casting para float
    convFLOAT =lambda x: float(x)
    df["taxaConclusao"] = df["taxaConclusao"].apply(convFLOAT)
    df["IRA"] = df["IRA"].apply(convFLOAT)
    df["totalpontos"] = df["totalpontos"].apply(convFLOAT)
    #coloca o ira e taxa de conclusao entre 0 e 1, apenas dividindo por 100:
    df["taxaConclusao"] = df["taxaConclusao"]/100
    df["IRA"] = df["IRA"]/100
    return df

#Recebe uma coluna do dataframe e a codifica por rótulos inteiros
#Está sendo usada para variáveis categóricas ordinais, transformando-as em quantitativas e discretas
from sklearn import preprocessing

def codificaCategoria(minhaserie):
    le = preprocessing.LabelEncoder()
    le.fit(minhaserie)
    #retorna a coluna codificada e o objeto le para converter os códigos em rótulos novamente.
    return le.transform(minhaserie), le


def criaColunaCodificada(df, colname, lstDecode):
    df[colname + 'CODE'], dec = codificaCategoria(df[colname])
    lstDecode.append(dec)
    return df


# Trata variáveis categóricas ordinais, transformando-as em quantitativas discretas
def trataCategoricasOrdinais(df):
    # armazena os objetos usados para decodificar as colunmas
    lstDecode = []
    # todo verificar cotas por ano.
    df = criaColunaCodificada(df, 'cota', lstDecode)
    # TODO FIXME
    # Temporariamente, usa-se essa codificação para ordinais em variáveis nominais.
    # depois será necessário implementar um tratamento mais adequado
    df = criaColunaCodificada(df, 'sexo', lstDecode)
    df = criaColunaCodificada(df, 'etnia', lstDecode)
    df = criaColunaCodificada(df, 'cota', lstDecode)
    df = criaColunaCodificada(df, 'grandearea', lstDecode)
    df = criaColunaCodificada(df, 'turno', lstDecode)
    df = criaColunaCodificada(df, 'processoseletivo', lstDecode)
    df = criaColunaCodificada(df, 'campus', lstDecode)
    df = criaColunaCodificada(df, 'turno', lstDecode)
    return df, lstDecode


# variavel global com as colunas consideradas para o estudo
colsIncluidas = ['Situacao', 'totalpontos', 'turno', 'sexo', 'etnia', 'cota', 'grandearea',
                 'anoconc2g', 'anoing', 'IRA', 'prazoideal', 'taxaConclusao', 'periodosCursados',
                 'processoseletivo', 'campus']


# esta função abre o dataset e realiza os tratamentos necessários para utilizar os algoritmos:
def criaDataFrame():
    df = pd.read_csv('esic12Set2020.csv', encoding='utf_8', sep=';')
    # filtra o dataframe para somente considerar os alunos da graduação presencial
    dfFiltro = df[(df["tipocurso"] == "GRADUAÇÃO PRESENCIAL")]
    # Pega apenas as colunas que serão utilizadas.
    dfEvasao = dfFiltro[colsIncluidas]
    # Dropa as linhas com as colunas nulas
    dfEvasao = dfEvasao.dropna()
    # calcula o GAP
    dfEvasao = calculaGAP(dfEvasao)
    # trata variáveis contínuas
    dfEvasao = trataQuantiContinuas(dfEvasao)
    # categoricas
    dfEvasao, lstDecode = trataCategoricasOrdinais(dfEvasao)
    return dfEvasao

#separa o data set com ativos e outro com concluidos e evadidos
def separaAtivos(df):
    #cria um dataframe somente com alunos evadidos ou concluidos para treinamento
    dfEvasaoSemAtivos = df[(df['Situacao']!='ATIVO')]
    #cria um dataframe somente com alunos ativos para depois classifica-los com evadido ou concluido
    dfEvasaoAtivos = df[(df['Situacao']=='ATIVO') ]
    return dfEvasaoSemAtivos, dfEvasaoAtivos

#calcula e imprime as proporcoes de evasao e conclusao
def proporcoes(nConc, nEvad):
    total = nConc + nEvad
    print("Concluidos: \t",nConc, " = ",nConc/total*100,"%")
    print("Evadidos: \t ",nEvad, " = ",nEvad/total*100,"%")
    print("Total: \t",  total)
#imprime estatistica na coluna de situação
def estatiscaSituacao(serieConcl):
    cont = serieConcl.value_counts()
    proporcoes(cont["CONCLUIDO"], cont["EVADIDO"])

from sklearn.model_selection import train_test_split
#colunas de entrada
#,'cotaCODE','etniaCODE', 'grandeareaCODE','turnoCODE', 'processoseletivoCODE',,'taxaConclusao','periodosCursados','totalpontos',
colsTreino = [ 'GAP', 'prazoideal', 'IRA', 'sexoCODE', 'campusCODE']
#separa o dataframe em treinamento e tese
def separaTreinoTeste(df):
    X = df[ colsTreino ]
    #somente a coluna com classificação (Situacao)
    Y = df['Situacao']
    #divide os dados
    return train_test_split(X, Y, test_size=0.7) # 70% training and 30% test

#recebe um objeto classificador e preve a classificacao para alunos ativos
def previsaoAtivos(objClassificador, dfEntradaAtivos):
    previsaoativos = objClassificador.predict(dfEntradaAtivos)
    s = pd.Series(previsaoativos)
    print("ARVORE: Estatítiscas da situação da classificação dos ativos")
    estatiscaSituacao(s)

def previsaoAtivosKNN(objClassificador, dfEntradaAtivos):
    previsaoativos = objClassificador.predict(dfEntradaAtivos)
    s = pd.Series(previsaoativos)
    print("KNN: Estatítiscas da situação da classificação dos ativos")
    estatiscaSituacao(s)


from sklearn.metrics import classification_report
from sklearn import metrics
#realiza a análise estatística entre previsoes e testes
def analisaDesempenho(y_pred, y_test):
    #compara as classificações calculadas pela árvore y_pred com a classificações original do dataset y_Test
    acuracia = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:",acuracia)
    report = classification_report(y_test,y_pred)
    #print(report)


from sklearn import tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt


# método que treina a árvore de decisao
def arvoreDecisao(X_train, y_train, X_test, y_test, dfEntradaAtivos):
    # cria um objeto classificador
    arvoreEvaso = tree.DecisionTreeClassifier()
    # Realiza o treinamento - X é o conjunto de entrada e Y é a respectiva classificação.
    arvoreTreinada = arvoreEvaso.fit(X_train, y_train)
    # Usa a arvore para classificar o dataset de test.
    y_arvore = arvoreTreinada.predict(X_test)
    previsaoAtivos(arvoreTreinada, dfEntradaAtivos)
    analisaDesempenho(y_arvore, y_test)
    print('\n')
    # desenha a árovre de decisao
    # plt.figure(figsize=(25,20))
    # tree.plot_tree(arvoreTreinada, filled=True, feature_names=colsTreino, max_depth=5, fontsize=15, class_names=True)
    # plt.show()

def knn(X_train, y_train, X_test, y_test, dfEntradaAtivos):
    #cria um novo classificador e treina com a base parcial
    knn = KNeighborsClassifier(n_neighbors=5)
    knntreinado= knn.fit(X_train,y_train)
    #realiza a previsao das entradas do teste
    pred = knntreinado.predict(X_test)
    previsaoAtivosKNN(knntreinado, dfEntradaAtivos)
    analisaDesempenho(pred, y_test)
    print('\n')
    #imprime as medidas


def inteligenciaartificial(dfEvasao, percConcPeriodos):
    # monta data frame conforme percentual de conclusao de periodos

    # filtra os dados para selecionar apenas alunos que cursaram mais da metade do prazo ideal
    dfEvasao = dfEvasao[(dfEvasao['periodosCursados'] > percConcPeriodos * dfEvasao['prazoideal'])]

    dfEvasaoConclusao, dfAtivos = separaAtivos(dfEvasao)
    X_treino, X_teste, y_treino, y_teste = separaTreinoTeste(dfEvasaoConclusao)
    print("Estatítiscas da situação da base de treinamento com ",percConcPeriodos*100,"% de conclusão")
    estatiscaSituacao(dfEvasaoConclusao['Situacao'])
    print("\n")
    arvore = arvoreDecisao(X_treino, y_treino, X_teste, y_teste, dfAtivos[colsTreino])
    # adiciona novas IAs aqui
    # KNN
    knneighbors= knn(X_treino, y_treino, X_teste, y_teste, dfAtivos[colsTreino])
    # etc..


dfEvasao = criaDataFrame()


#treina com 50% ou mais de conclusão
inteligenciaartificial(dfEvasao, 0.5)
inteligenciaartificial(dfEvasao, 0.4)
inteligenciaartificial(dfEvasao, 0.3)
inteligenciaartificial(dfEvasao, 0.2)
