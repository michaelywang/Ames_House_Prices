def process(train,test):
    import pandas as pd
    pd.set_option('display.max_columns', 500)
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')


    train = pd.read_csv(f'{train}.csv')
    test = pd.read_csv(f'{test}.csv')

    saved_train = train[['Id','SalePrice']]
    train = train.drop(columns='SalePrice')
    df = pd.concat([train,test])

        #extracting data and creating data frames for transformation and merge
    train_df_4_neigh_avg = pd.read_csv('train.csv')
    avg_df = train_df_4_neigh_avg.groupby(['Neighborhood']).agg({'SalePrice':'mean'}).reset_index().sort_values('SalePrice')
    avg_df['SalePrice'] = avg_df['SalePrice'].round(decimals=0)

    #creating 5 bins for avg sale price, looking at distribution of total neighborhoods by bin
    avg_df_bins = pd.DataFrame(pd.cut(avg_df['SalePrice'],20)).groupby('SalePrice').agg({'SalePrice':'count'})
    avg_df_bins.columns = ["_".join(x) for x in avg_df_bins.columns.ravel()]
    avg_df_bins = avg_df_bins.reset_index()
    avg_df_bins.rename(columns={ avg_df_bins.columns[1]: "Neighborhoods" }, inplace = True)

    #add column for value to be added to bin
    bin_val = pd.DataFrame({"Neighborhood_binVal":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]})
    avg_df_bins = pd.concat([avg_df_bins,bin_val],axis=1)
    avg_df_bins.rename(columns={'SalePrice':'Bin_Sale'}, inplace=True)

    #recreating object containing bins - why? so that we can join on Bin_Sale for neighborhood bin values
    cuts = pd.DataFrame(pd.cut(avg_df['SalePrice'],20))
    cuts.rename(columns={'SalePrice':'Bin_Sale'}, inplace=True)
    neigh_bins = pd.concat([cuts, avg_df], axis=1)
    neigh_bins.drop(columns=neigh_bins.columns[2],inplace=True)
    neigh_bins = pd.merge(left=avg_df_bins, right=neigh_bins, how='inner', on='Bin_Sale')
    neigh_bins.drop(columns=neigh_bins.columns[0:2], inplace=True)

    #merge neighborhood value bins with dataset
    df = pd.merge(left=df, right=neigh_bins, on='Neighborhood', how='inner')

    #Replacing old Neighborhood column with new version featuring new values based on sale price bin neighborhood fell within
    df.drop(columns={'Neighborhood'}, inplace=True)
    df.rename(columns={'Neighborhood_binVal':'Neighborhood'}, inplace=True)

    df_vix = pd.read_csv('vixcurrent.csv',skiprows=1)
    vix_date = df_vix['Date'].str.split("/", n = 2, expand=True)
    df_vix['Month'] = vix_date[0].astype('int')
    df_vix['Day'] = vix_date[1].astype('int')
    df_vix['Year'] = vix_date[2].astype('int')
    df_vix = df_vix.drop(columns='Date')
    vix_mo_yr = df_vix.groupby(by=['Year','Month']).agg({'Day':'min','VIX Close':'mean'}).reset_index().rename(columns={'VIX Close':'VIX_Close_Mo_Avg'})
    df_vix_to_append = pd.merge(left=df_vix, right=vix_mo_yr, how='inner', on=['Year','Month','Day'])
    df_vix_to_append = df_vix_to_append[['Month','Year','VIX_Close_Mo_Avg']]
    df = pd.merge(left=df, right=df_vix_to_append, how='inner', left_on=['MoSold','YrSold'], right_on=['Month','Year'])


    coldrop = ['YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GarageYrBlt']
    df = df.drop(columns=coldrop)

    df['LotFrontage'] = df['LotFrontage'].fillna(np.mean(df['LotFrontage']))
    df['MasVnrArea'].fillna(0,inplace=True)

    df['BsmtFullBath'] = df[['BsmtHalfBath']].apply(lambda x: (x*.5)+df['BsmtFullBath'])
    df.drop('BsmtHalfBath',axis=1,inplace=True) # dropping basment half bath after combining total with full
    df.rename(columns = {'BsmtFullBath':'BsmtBath'},inplace=True) # renaming bath column
    df['FullBath'] = df[['HalfBath']].apply(lambda x: (x*.5)+df['FullBath'])
    df.drop('HalfBath',axis=1,inplace=True) # dropping half bath after combining total with full
    df.rename(columns = {'Bath':'Bath'},inplace=True) # renaming bath column

    colnames= ['ExterQual', 'ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual', 'GarageCond']
    toreplace=['Ex','Gd','TA','Fa','Po',np.nan]
    replacewith= [5,4,3,2,1,0]
    for i in df[colnames]:
        df[i] = df[i].replace(toreplace,replacewith)
# this should leave us with 22 variables to dummify, and with the above code, we should have 17 int64 columns
    df['Utilities'] = df['Utilities'].replace(['AllPub','NoSewr','NoSeWa','ELO'],[4,3,2,1])
    df['LandSlope'] = df['LandSlope'].replace(['Gtl','Mod','Sev'],[3,2,1])
    df['BsmtExposure'] = df['BsmtExposure'].replace(['Gd','Av','Mn','No',np.nan],[4,3,2,1,0])
    df['BsmtFinType1'] = df['BsmtFinType1'].replace(['GLQ','ALQ','BLQ','Rec','LwQ','Unf',np.nan],[6,5,4,3,2,1,0])
    df['BsmtFinType2'] = df['BsmtFinType2'].replace(['GLQ','ALQ','BLQ','Rec','LwQ','Unf',np.nan],[6,5,4,3,2,1,0])
    df['Functional'] = df['Functional'].replace(['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],[1,2,3,4,5,6,7,8])
    df['GarageFinish'] = df['GarageFinish'].replace(['Fin','RFn','Unf',np.nan],[3,2,1,0])
    df['PavedDrive'] = df['PavedDrive'].replace(['Y','P','N'],[3,2,1])
    df['CentralAir'] = df['CentralAir'].replace(['Y','N'],(1,0))
    df['Street'] = df['Street'].replace(['Pave','Grvl'],(1,0))

    df["BsmtBath"].fillna(0, inplace=True)
    df["TotalBsmtSF"].fillna(0, inplace=True)
    df["Functional"].fillna(8, inplace=True)
    df['GarageArea'].fillna(0,inplace=True)
    df['GarageCars'].fillna(0,inplace=True)
    df["Utilities"].fillna(4.0,inplace=True)
    df.drop(columns=['Alley','PoolQC'],inplace=True)
    # make sure there are 22 columns in this list
    df.select_dtypes(include='object').columns

    dummies = list(df.select_dtypes(include='object').columns)

    dum_df = pd.get_dummies(df[dummies])

    df = pd.concat([df,dum_df],axis=1,join='outer')
    df = pd.merge(df,saved_train,how='left',on='Id')

    df = df.sort_values(by='Id').reset_index()
    df.drop('index',axis = 1,inplace=True)
    df.drop(columns=dummies,inplace=True)

    # remember for the below that it won't necessarily be true for the test
    dum_drops = ['MSZoning_RL', 'LotShape_Reg', 'LandContour_Lvl', 'LotConfig_Inside','Condition1_Norm', 'Condition2_Norm', 'BldgType_1Fam', 'HouseStyle_1Story', 'RoofStyle_Gable','RoofMatl_CompShg', 'Exterior1st_VinylSd', 'Exterior2nd_VinylSd', 'MasVnrType_None', 'Foundation_PConc','Heating_GasA', 'Electrical_SBrkr', 'GarageType_Attchd', 'Fence_MnPrv', 'MiscFeature_Shed','SaleType_WD', 'SaleCondition_Normal']
    df.drop(columns=dum_drops,inplace=True)
    df.drop(columns='Id',inplace=True)


    return df
