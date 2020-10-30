import pandas as pd
from sklearn import tree
from sklearn.tree import export_text
from matplotlib import pyplot as plt
df = pd.read_csv('iris.csv', engine="python", index_col=False, header=0, encoding='utf-8-sig')
X = df[['AltSepala', 'LargSepala', 'AlturaPetala','LarguraPetala']]
Y = df['Especie']
arvoreIris = tree.DecisionTreeClassifier()
classificaIris = arvoreIris.fit(X,Y)
caracteristicas = list(df.columns[0:4])
r = export_text(classificaIris, feature_names=caracteristicas)
print(r)

plt.figure(figsize=(10,10))
tree.plot_tree(classificaIris, filled=True, feature_names=caracteristicas,class_names=['Setosa','Versicolour','Virginica'])
plt.show()