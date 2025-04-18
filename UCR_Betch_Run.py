#!/usr/bin/env python
# coding: utf-8

# In[324]:


import pandas as pd
import os
from DataLoad_Utils.TS_DataLoader import load_from_tsfile_to_dataframe, sort_timeseries
from DataLoad_Utils.My_Datautils import (Draw_Visiual_Multi_Feature_With_ZNormalization, Draw_Visiual_Multi_Feature, Draw_Visiual_Single_Feature_Each_Picture,
                                         load_UCR, remove_after_dot, Draw_Visiual_Single_Feature, plot_features_with_selector, read_tsf_file, CDF_Visual, CDF_Visual_Save,
                                         PDF_Visual, ACF_Visual, PACF_Visual, PSD_Visual, Hurst_Exponent_Visual)
from DataLoad_Utils.TSF_DataLoader import convert_tsf_to_dataframe, convert_tsf_to_dataframea_array
from DataLoad_Utils.TSF import TSF_Draw_Visiual_Single_Feature, TSF_Draw_Visiual_Single_Feature_Each_Picture, TSF_Draw_Visiual_Multi_Feature
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

# In[325]:


# data_dir = 'datasets/monash_datasets/australian_electricity_demand_dataset/australian_electricity_demand_dataset.tsf'

# data_dir = 'DiatomSizeReduction'
# data_dir = 'datasets/data/energy_data.csv'
# data_dir = 'datasets/ETT-small/ETTh1.csv'
# data_dir = 'datasets/data/energydata_complete.csv'
data_dir = 'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AppliancesEnergy/AppliancesEnergy_TEST.ts'

'''
datasets = [
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "GunPoint",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
    "ACSF1",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "BME",
    "Chinatown",
    "Crop",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "MelbournePedestrian",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "PowerCons",
    "Rock",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "SmoothSubspace",
    "UMD"
]
'''

# datasets = [
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AppliancesEnergy/AppliancesEnergy.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AustraliaRainfall/AustraliaRainfall.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BeijingPM10Quality/BeijingPM10Quality.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality/BeijingPM25Quality.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BenzeneConcentration/BenzeneConcentration.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32HR/BIDMC32HR.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32RR/BIDMC32RR.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32SpO2/BIDMC32SpO2.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/Covid3Month/Covid3Month.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling1/FloodModeling1.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling2/FloodModeling2.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling3/FloodModeling3.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/HouseholdPowerConsumption1/HouseholdPowerConsumption1.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/HouseholdPowerConsumption2/HouseholdPowerConsumption2.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/IEEEPPG/IEEEPPG.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/LiveFuelMoistureContent/LiveFuelMoistureContent.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/NewsHeadlineSentiment/NewsHeadlineSentiment.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/NewsTitleSentiment/NewsTitleSentiment.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/PPGDalia/PPGDalia.ts'
# ]

# datasets = [
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AppliancesEnergy/AppliancesEnergy_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AustraliaRainfall/AustraliaRainfall_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BeijingPM10Quality/BeijingPM10Quality_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality/BeijingPM25Quality_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BenzeneConcentration/BenzeneConcentration_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32HR/BIDMC32HR_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32RR/BIDMC32RR_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32SpO2/BIDMC32SpO2_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/Covid3Month/Covid3Month_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling1/FloodModeling1_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling2/FloodModeling2_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling3/FloodModeling3_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/HouseholdPowerConsumption1/HouseholdPowerConsumption1_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/HouseholdPowerConsumption2/HouseholdPowerConsumption2_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/IEEEPPG/IEEEPPG_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/LiveFuelMoistureContent/LiveFuelMoistureContent_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/NewsHeadlineSentiment/NewsHeadlineSentiment_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/NewsTitleSentiment/NewsTitleSentiment_TRAIN.ts',
#     'datasets/TSER-Monash_UEA_UCR_Regression_Archive/PPGDalia/PPGDalia_TRAIN.ts'
# ]

datasets = [
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AppliancesEnergy/AppliancesEnergy_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/AustraliaRainfall/AustraliaRainfall_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BeijingPM10Quality/BeijingPM10Quality_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BeijingPM25Quality/BeijingPM25Quality_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BenzeneConcentration/BenzeneConcentration_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32HR/BIDMC32HR_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32RR/BIDMC32RR_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/BIDMC32SpO2/BIDMC32SpO2_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/Covid3Month/Covid3Month_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling1/FloodModeling1_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling2/FloodModeling2_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/FloodModeling3/FloodModeling3_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/HouseholdPowerConsumption1/HouseholdPowerConsumption1_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/HouseholdPowerConsumption2/HouseholdPowerConsumption2_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/IEEEPPG/IEEEPPG_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/LiveFuelMoistureContent/LiveFuelMoistureContent_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/NewsHeadlineSentiment/NewsHeadlineSentiment_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/NewsTitleSentiment/NewsTitleSentiment_TEST.ts',
    'datasets/TSER-Monash_UEA_UCR_Regression_Archive/PPGDalia/PPGDalia_TEST.ts'
]




# In[326]:


# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--Save_OR_Not', default=True, type=bool, help='是否保存结果图')
# parser.add_argument('--cut_OR_Not', default=True, type=bool, help='是否对特征列超过30的数据进行截断')
# args = parser.parse_args(args=['--device', '0',  '--no_cuda'])
# print(args)


# In[327]:

for data_dir in datasets:

    file_type = None
    if '/' in data_dir:
        ## parameters
        file_name = os.path.basename(data_dir)
        result_dir, file_type = remove_after_dot(data_dir)  # 截取文件夹路径和文件类型
    else:
        file_name = data_dir
        result_dir = "monash/" + file_name

    parameters = dict()
    parameters['Save_OR_Not'] = True  # 是否保存结果图
    parameters['cut_OR_Not'] = True  # 是否对特征列过长的数据进行截断
    parameters['cut_length'] = 30  # 截断长度
    parameters['CDF_OR_Not'] = False  # 是否显示CDF图
    parameters['file_name'] = file_name

    # In[328]:

    if file_type == 'tsf':  # monash_datasets
        # df= read_tsf_file(data_dir)
        df, _, _, _, _ = convert_tsf_to_dataframea_array(data_dir)
        # df,_, _, _, _  = convert_tsf_to_dataframe(data_dir)
    elif file_type == 'ts':  # TSER-Monash_UEA_UCR_Regression_Archive
        # # set data folder, train & test
        # data_folder = data_dir + file_name + "/"
        # train_file = result_dir + "_TRAIN.ts"
        # test_file = result_dir + "_TEST.ts"
        train_file = data_dir
        module = "RegressionExperiment"
        # loading the data. X_train and X_test are dataframe of N x n_dim
        print("[{}] Loading data".format(module))
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        # X_test, y_test = load_from_tsfile_to_dataframe(test_file)
        # df1 = X_train
        # df1['dim_other'] = y_train
        # df2 = X_test
        # df2['dim_other'] = y_test
        # df = pd.concat([df1, df2], ignore_index=True)
        df = X_train
        df['dim_other'] = y_train
        df = sort_timeseries(df)
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'date'})
        if 'Index' in df.columns:
            df = df.drop('Index', axis=1)

    elif file_type == 'csv' or file_type == 'txt':
        df = pd.read_csv(data_dir)
    elif '/' not in data_dir:
        df, _ = load_UCR(data_dir)
    else:
        raise ValueError('Invalid file type')

    json_dir = result_dir
    # 保存图的路径
    result_dir = 'Result/' + result_dir
    print(result_dir)

    # In[329]:

    # df=df.iloc[20000:, :2]
    print(df.shape)
    columns_list = df.columns.tolist()
    print(columns_list)

    # In[330]:

    df
    """
    series_name	state	start_timestamp	series_value
    T1	NSW	2002-01-01	[5714.045004, 5360.189078, 5014.835118,4602.755516, 4285.179828,4074.8...]
    T2	VIC	2002-01-01	[3535.867064,3383.499028,3655.527552,3510.446636,3294.697156,3111.8...]
    T3	QUN	2002-01-01	[3382.041342,3288.315794,3172.329022,3020.312986,2918.082882, 2839.9...]
    T4	SA	2002-01-01	[1191.078014, 1219.589472, 1119.173498, 1016.407248, 923.499578, 855.867...]
    T5	TAS	2002-01-01	[315.915504,306.245864,305.762576,295.602196, 290.44707, 282.57759, 2...]

    """

    # In[331]:

    from DataLoad_Utils.Data_CDF import analyze_dataset, save_analysis_to_json, analyze_dataset_tsf, analysis_to_excel

    if file_type == 'tsf':
        # 执行分析
        tsf_analysis = analyze_dataset_tsf(df, file_name)
        analysis_to_excel(tsf_analysis, f"{result_dir + '/' + file_name}.xlsx")
        # 保存结果
        # save_analysis_to_json([tsf_analysis], f"JSON/{json_dir}.json")

    elif file_type == 'csv' or file_type == 'txt' or file_type == 'ts':
        csv_analysis = analyze_dataset(df, file_name)
        analysis_to_excel(csv_analysis, f"{result_dir + '/' + file_name}.xlsx")
        # save_analysis_to_json([csv_analysis], f"JSON/{json_dir}.json")

    elif '/' not in data_dir:
        UCR_analysis = analyze_dataset(df, file_name)
        analysis_to_excel(UCR_analysis, f"{result_dir + '/' + file_name}.xlsx")
        # save_analysis_to_json([UCR_analysis], f"JSON/{json_dir}.json")

    # In[332]:

    # from TSF_Visual import plot_time_series

    # if file_type == 'tsf':
    #     TSF_Draw_Visiual_Single_Feature(df, columns_list, parameters, result_dir)
    # else:
    #     Draw_Visiual_Single_Feature(df, columns_list, parameters, result_dir)

    # In[333]:

    if file_type == 'tsf':
        TSF_Draw_Visiual_Single_Feature_Each_Picture(df, columns_list, parameters, result_dir)
    else:
        Draw_Visiual_Single_Feature_Each_Picture(df, columns_list, parameters, result_dir)

    # In[334]:

    if file_type == 'tsf':
        TSF_Draw_Visiual_Multi_Feature(df, columns_list, parameters, file_name, result_dir)
    else:
        Draw_Visiual_Multi_Feature(df, columns_list, parameters, result_dir)

    # In[335]:

    # CDF_Visual(df, columns_list, parameters)
    CDF_Visual_Save(df, columns_list, parameters, result_dir)

    # In[336]:

    # PDF_Visual(df, columns_list, parameters)

    # In[337]:

    # ACF_Visual(df, columns_list, parameters)

    # In[338]:

    # PACF_Visual(df, columns_list, parameters)

    # In[339]:

    # PSD_Visual(df, columns_list, parameters)

    # In[ ]:

    # In[340]:

    # Hurst_Exponent_Visual(df, columns_list, parameters)

    # In[341]:

    # Draw_Visiual_Multi_Feature_With_ZNormalization(df, columns_list, parameters, result_dir)

    # In[342]:

    # plot_features_with_selector(df, columns_list, Save_OR_Not, result_dir) #<---------
    # # plot_features_with_selector(df, df.columns.tolist())
    # # plot_features_horizontal_selector(df, df.columns.tolist())

    # In[ ]:




