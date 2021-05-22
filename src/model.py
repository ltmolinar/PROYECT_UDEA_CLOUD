from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as  np
import pickle
from azureml.core import Dataset
from azureml.core import Workspace

if __name__ == "__main__":
        
        df = pd.read_csv(filepath_or_buffer='data.csv', delimiter=';')

    #ws = Workspace.from_config(path='../.azureml',_file_name='config.json')
    #datastore = ws.get_default_datastore()
    #df = Dataset.File.from_files(path=(datastore, 'datasets/muestradatos'))


        #typo de datos adecuados para codificación de varibales categoricas
        df['descTema'] = df['descTema'].astype('str')
        df['descCanalRadicacion'] = df['descCanalRadicacion'].astype('str')
        df['descSegmentoAfiliado'] = df['descSegmentoAfiliado'].astype('str')
        df['descCicloVida'] = df['descCicloVida'].astype('str')
        df['descOcupacion'] = df['descOcupacion'].astype('str')
        df['descRegional'] = df['descRegional'].astype('str')

        #codificación de variables:
        LE = LabelEncoder()
        df2 = df[['afi_hash64','descTema','descSexo', 'descSegmentoAfiliado', 'edadAfiliado', 
                'EstadoPO', 'EstadoPV', 'EstadoCES', 'ultimoIBC','IndicadorUsaClave', 'idAfiliadoTieneClave', 'TieneEmail',
                'descCicloVida','descOcupacion', 'descRegional' , 'descCanalRadicacion']]

        df2 = df2[df2.descCanalRadicacion.isin(['LINEA DE SERVICIO', 'OFICINA DE SERVICIO', 'OFICINA VIRTUAL'])]

        df2['afi_hash64'] = LE.fit_transform(df2['afi_hash64'])
        df2["descTema"] = LE.fit_transform(df2['descTema'])
        df2["descSexo"] = LE.fit_transform(df2['descSexo'])
        df2["descSegmentoAfiliado"] = LE.fit_transform(df2['descSegmentoAfiliado'])
        df2["EstadoPO"] = LE.fit_transform(df2['EstadoPO'])
        df2["EstadoPV"] = LE.fit_transform(df2['EstadoPV'])
        df2["EstadoCES"] = LE.fit_transform(df2['EstadoCES'])
        df2["descCanalRadicacion"] = LE.fit_transform(df2['descCanalRadicacion'])
        df2["IndicadorUsaClave"] = LE.fit_transform(df2['IndicadorUsaClave'])
        df2["idAfiliadoTieneClave"] = LE.fit_transform(df2['idAfiliadoTieneClave'])
        df2["descCicloVida"] = LE.fit_transform(df2['descCicloVida'])
        df2["TieneEmail"] = LE.fit_transform(df2['TieneEmail'])
        df2["descOcupacion"] = LE.fit_transform(df2['descOcupacion'])
        df2["descRegional"] = LE.fit_transform(df2['descRegional'])

        # Eliminamos los registros de clientes que de acuerdo con el conocimiento del negocio no nos contactan normalmente

        df3 = df2.drop(df.index[ (df['edadAfiliado'] > 90) & (df['edadAfiliado'] < 18)])

        #normalización de varibles:
        scaler = MinMaxScaler()
        scaler.fit(df3.iloc[:,1:15])
        df4= scaler.transform(df3.iloc[:,1:15])
        df4 = pd.DataFrame(df4, columns =  ['descTema','descSexo', 'descSegmentoAfiliado', 'edadAfiliado', 
                'EstadoPO', 'EstadoPV', 'EstadoCES', 'ultimoIBC','IndicadorUsaClave', 'idAfiliadoTieneClave', 
                'TieneEmail', 'descCicloVida','descOcupacion', 'descRegional' ])

        #Separación train y test
        
        X_train, X_test, y_train, y_test = train_test_split(df4,df3["descCanalRadicacion"], 
                test_size=0.2, random_state=10, stratify =df3["descCanalRadicacion"] )

        #ML Model: Model Selection
        knn = KNeighborsClassifier(n_neighbors=50, weights='distance')

        sfs1 = SFS(knn, 
                k_features=11, 
                forward=True, 
                floating=False, 
                verbose=1,
                scoring= make_scorer(f1_score, average = 'weighted'),
                cv=5)

        sfs1 = sfs1.fit(X_train, y_train)

        X_train_sfs = sfs1.transform(X_train)
        X_test_sfs = sfs1.transform(X_test)

        clfKnn_sfs = knn.fit(X_train_sfs, y_train)

        #Predictions
        predictions=clfKnn_sfs.predict(X_test_sfs)

        #Valor F
        print(f1_score(y_test,predictions.astype('int64'), average='weighted'))

        #Registro
        with open('./outputs/model_uso_canales.pkl', 'wb') as model_pkl:
                pickle.dump(knn, model.pkl)