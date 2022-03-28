Seu conjunto de dados tinha muitas variáveis para entender ou até mesmo para imprimir bem. Como você pode reduzir essa enorme quantidade de dados para algo que você possa entender?
**Começaremos escolhendo algumas variáveis usando nossa intuição**. **Os cursos posteriores mostrarão técnicas estatísticas para priorizar variáveis automaticamente.**
Para escolher variáveis/colunas, precisaremos ver uma lista de todas as colunas no conjunto de dados. Isso é feito com a propriedade columns do DataFrame (a linha inferior do código abaixo).

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a67e23a-d6a5-439d-90cc-8b40144b8ac3/Untitled.png)

```python
# Os dados de Melbourne têm alguns valores ausentes (algumas
# casas para as quais algumas variáveis não foram registradas).
# Aprenderemos a lidar com valores ausentes em um tutorial posterior.
# Seus dados de Iowa não têm valores ausentes nas colunas que você usa.
# Então, vamos pegar a opção mais simples por enquanto e descartar as casas
# dos nossos dados.
# Não se preocupe muito com isso por enquanto, embora o código seja:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```

Há muitas maneiras de selecionar um subconjunto de seus dados. O curso Pandas cobre isso com mais profundidade, mas vamos nos concentrar em duas abordagens por enquanto.

**Notação de ponto, que usamos para selecionar o "alvo de previsão"**
**Selecionando com uma lista de colunas, que usamos para selecionar os "recursos"**

# **Selecionando o alvo de previsão(Selecting The Prediction Target)**

Você pode extrair uma variável com notação de ponto. Essa única coluna é armazenada em uma série, que é amplamente semelhante a um DataFrame com apenas uma única coluna de dados.
**Usaremos a notação de ponto para selecionar a coluna que queremos prever**, que é chamada de **destino de previsão**. **Por convenção, a meta de previsão é chamada de y**. Portanto, o código que precisamos para salvar os preços das casas nos dados de Melbourne é

```python
y = melbourne_data.Price
```

# ****Choosing "Features"(escolhendo recursos)****

**As colunas que são inseridas em nosso modelo (e posteriormente usadas para fazer previsões) são chamadas de "features"**. No nosso caso, essas seriam as colunas **usadas para determinar o preço da casa**. Às vezes, você usará todas as colunas, exceto o destino, como recursos. **Outras vezes, você ficará melhor com menos recursos.**

Por enquanto, vamos construir um modelo com apenas alguns recursos. Mais tarde, você verá como iterar e comparar modelos criados com diferentes recursos.

**Selecionamos vários recursos fornecendo uma lista de nomes de colunas entre colchetes. Cada item dessa lista deve ser uma string (com aspas).**

Aqui está um exemplo:

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```

Por convenção, esses dados são chamados de X.

```python
X = melbourne_data[melbourne_features]
```

Vamos revisar rapidamente os dados que usaremos para prever os preços das casas usando o método describe e o método head, que mostra as primeiras linhas.

```python
X.describe()
```

```python
X.head()
```

A verificação visual de seus dados com esses comandos é uma parte importante do trabalho de um cientista de dados**. Você encontrará frequentemente surpresas no conjunto de dados que merecem uma inspeção mais detalhada.**

# **Construindo seu modelo**

Você usará a biblioteca **scikit-learn** para criar seus modelos. Ao codificar, esta biblioteca é escrita como sklearn, como você verá no código de exemplo. Scikit-learn é facilmente a biblioteca mais popular para modelar os tipos de dados normalmente armazenados em DataFrames.

As etapas para construir e usar um modelo são:

**Defina: Que tipo de modelo será? Uma árvore de decisão?** Algum outro tipo de modelo? Alguns outros parâmetros do tipo de modelo também são especificados.
Ajuste: **capture padrões dos dados fornecidos**. Este é o coração da modelagem.
Prever: **exatamente o que parece**
Avaliar: **determine a precisão das previsões do modelo.**
Aqui está um exemplo de definição de um modelo de árvore de decisão com scikit-learn e ajustá-lo com os recursos e a variável de destino.

```python
from sklearn.tree import DecisionTreeRegressor # importa a biblioteca

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1) # dimensao da arvore

# Fit model
melbourne_model.fit(X, y)

####  DecisionTreeRegressor(random_state=1)
```

muitos modelos de aprendizado de máquina permitem alguma aleatoriedade no treinamento do modelo. **Especificar um número para random_state garante que você obtenha os mesmos resultados em cada execução. Isso é considerado uma boa prática. Você usa qualquer número e a qualidade do modelo não dependerá significativamente do valor escolhido**.

Agora temos um modelo ajustado que podemos usar para fazer previsões.

Na prática, você desejará fazer previsões para novas casas no mercado, em vez das casas para as quais já temos preços. Mas faremos previsões para as primeiras linhas dos dados de treinamento para ver como a função de previsão funciona.

```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

#dados de treino 
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/912cdabb-13cf-4e42-9e09-a8802be8be45/Untitled.png)

# ****Model Validation****

**Você construiu um modelo. Mas quão bom é?
Nesta lição, você aprenderá a usar a validação de modelo para medir a qualidade do seu modelo.** Medir a qualidade do modelo é a chave para melhorar seus modelos de forma iterativa.

# O que é validação de modelo

Você vai querer avaliar quase todos os modelos que você já construiu. Na maioria das aplicações (embora não em todas), a medida relevante da qualidade do modelo é a precisão preditiva. **Em outras palavras, as previsões do modelo estarão próximas do que realmente acontece.**
Muitas pessoas cometem um grande erro ao medir a precisão preditiva. **Eles fazem previsões com seus dados de treinamento e comparam essas previsões com os valores de destino nos dados de treinamento.** Você verá o problema com essa abordagem e como resolvê-lo em breve, mas vamos pensar em como faríamos isso primeiro.
Primeiro, você precisa resumir a qualidade do modelo de forma compreensível. Se você comparar os valores de casas previstos e reais para 10.000 casas, provavelmente encontrará uma mistura de previsões boas e ruins. Examinar uma lista de 10.000 valores previstos e reais seria inútil. Precisamos resumir isso em uma única métrica.
**Existem muitas métricas para resumir a qualidade do modelo, mas começaremos com uma chamada Erro Médio Absoluto (também chamada MAE).** Vamos detalhar essa métrica começando com a última palavra, erro.
O erro de previsão para cada casa é:

```python
error=actual−predicted
```

**Então, se uma casa custa $ 150.000 e você previu que custaria $ 100.000, o erro é $ 50.000.**
Com a métrica MAE, tomamos o valor absoluto de cada erro. **Isso converte cada erro em um número positivo. Em seguida, tomamos a média desses erros absolutos**. Esta é a nossa medida de qualidade do modelo. Em linguagem simples, pode-se dizer como
Em média, nossas previsões estão erradas em cerca de X.
Para calcular o MAE, primeiro precisamos de um modelo.

```python
# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
```

**Uma vez que temos um modelo, aqui está como calculamos o erro absoluto médio:**

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

## ****The Problem with "In-Sample" Scores****

A medida que acabamos de calcular pode ser chamada de pontuação "in-sample". Usamos uma única "amostra" de casas tanto para construir o modelo quanto para avaliá-lo. Eis por que isso é ruim.
**Imagine que, no grande mercado imobiliário, a cor da porta não tem relação com o preço da casa.
No entanto, na amostra de dados que você usou para construir o modelo, todas as casas com portas verdes eram muito caras.**  
Como esse padrão foi derivado dos dados de treinamento, o modelo parecerá preciso nos dados de treinamento.
Mas se esse padrão não se mantiver quando o modelo vir novos dados, o modelo será muito impreciso quando usado na prática.
Como o valor prático dos modelos vem de fazer previsões em novos dados, medimos o desempenho em dados que não foram usados para construir o modelo. **A maneira mais direta de fazer isso é excluir alguns dados do processo de construção do modelo e usá-los para testar a precisão do modelo em dados que ele não viu antes. Esses dados são chamados de dados de validação.**

Codificando
A biblioteca **scikit**-**learn** tem uma função **train_test_split** para dividir os dados em duas partes. Usaremos alguns desses dados como dados de treinamento para ajustar o modelo e usaremos os outros dados como dados de validação para calcular o erro_absoluto médio.
Aqui está o código:

```python
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

####### saidaaa ############
258930.03550677857
```

Seu erro absoluto médio para os dados na amostra foi de cerca de 500 dólares. Fora da amostra é mais de 250.000 dólares.
Esta é a diferença entre um modelo que é quase exatamente correto e um que é inutilizável para a maioria dos propósitos práticos**. Como ponto de referência, o valor médio da casa nos dados de validação é de 1,1 milhão de dólares. Portanto, o erro em novos dados é cerca de um quarto do valor médio da casa.
Há muitas maneiras de melhorar esse modelo, como experimentar para encontrar recursos melhores ou diferentes tipos de modelo.**

# **Experimentando com diferentes modelos**

Agora que você tem uma maneira confiável de medir a precisão do modelo, pode experimentar modelos alternativos e ver qual fornece as melhores previsões. Mas quais alternativas você tem para os modelos?
Você pode ver na documentação do scikit-learn que o modelo de árvore de decisão tem muitas opções (mais do que você deseja ou precisa por um longo tempo). **As opções mais importantes determinam a profundidade da árvore. Lembre-se da primeira lição deste curso que a profundidade de uma árvore é uma medida de quantas divisões ela faz antes de chegar a uma previsão. Esta é uma árvore relativamente rasa**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f3fd987-a0ab-4582-86ca-8b40f7119c5c/Untitled.png)

**Na prática, não é incomum que uma árvore tenha 10 divisões entre o nível superior** (todas as casas) e uma folha. À medida que a árvore se aprofunda, o conjunto de dados é dividido em folhas com menos casas. Se uma árvore teve apenas 1 divisão, ela divide os dados em 2 grupos. Se cada grupo for dividido novamente, obteremos 4 grupos de casas. Dividir cada um deles novamente criaria 8 grupos. Se continuarmos dobrando o número de grupos adicionando mais divisões em cada nível, teremos 210 grupos de casas quando chegarmos ao 10º nível. São 1024 folhas.

Quando dividimos as casas entre muitas folhas, também temos menos casas em cada folha. **Folhas com muito poucas casas farão previsões bastante próximas dos valores reais dessas casas**, **mas podem fazer previsões muito pouco confiáveis para novos dados (porque cada previsão é baseada em apenas algumas casas).**

**Esse é um fenômeno chamado overfitting**, **em que um modelo corresponde aos dados de treinamento quase perfeitamente, mas se sai mal na validação e em outros novos dados**. Por outro lado, se fizermos nossa árvore muito rasa, ela não dividirá as casas em grupos muito distintos.

Em um extremo, se uma árvore divide as casas em apenas 2 ou 4, cada grupo ainda tem uma grande variedade de casas. As previsões resultantes podem estar distantes para a maioria das casas, mesmo nos dados de treinamento (e também serão ruins na validação pelo mesmo motivo). **Quando um modelo falha em capturar distinções e padrões importantes nos dados, ele apresenta um desempenho ruim mesmo nos dados de treinamento, isso é chamado de underfitting.**

Como nos preocupamos com a precisão em novos dados, que estimamos a partir de nossos dados de validação, queremos encontrar o ponto ideal entre underfitting e overfitting. Visualmente, queremos o ponto baixo da curva de validação (vermelha) na figura abaixo.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/459ca536-9a76-49ef-9d3b-0ec3e3bfbe3b/Untitled.png)

# Exemplo

Existem algumas alternativas para controlar a profundidade da árvore, e muitas permitem que algumas rotas através da árvore tenham maior profundidade do que outras rotas. **Mas o argumento max_leaf_nodes fornece uma maneira muito sensata de controlar overfitting vs underfitting**. Quanto mais folhas permitimos que o modelo faça, mais nos movemos da área de underfitting no gráfico acima para a área de overfitting.

**Podemos usar uma função de utilitário para ajudar a comparar as pontuações MAE de diferentes valores para max_leaf_nodes:**

```python
from sklearn.metrics importmean_absolute_error
from sklearn.tree import DecisionTreeRegressor 

def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):

	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
	model.fit(train_X,train_y)

	preds_val = model.predict(val_X)
	mae = mean_absolute_error(val_y,preds_val)
  return(mae)
```

Os dados são carregados em train_X, val_X, train_y e val_y usando o código que você já viu (e que já escreveu).

Olhe abaixo:

```python
# Data Loading Code Runs At This Point
import pandas as pd
    
# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```

Podemos usar um loop for para comparar a precisão de modelos construídos com valores diferentes para max_leaf_nodes.

```python
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5f4815a-ee53-49eb-98d0-098e822a38c7/Untitled.png)

## Conclusão

Aqui está o takeaway: Os modelos podem sofrer de:

Overfitting: captura de padrões espúrios que não se repetirão no futuro, levando a previsões menos precisas ou
Underfitting: falha em capturar padrões relevantes, novamente levando a previsões menos precisas.
Usamos dados de validação, que não são usados no treinamento de modelos, para medir a precisão de um modelo candidato. Isso nos permite experimentar muitos modelos candidatos e manter o melhor.

