
####
########## 1#####################
## Set up the server:
#### cd ..
#### cd Results
#### mkdir Step1
#### Then you can run this file.
########## 1#####################



# Importing necessary packages.
import os
import zipfile as zip
import pandas as pd
import csv
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

# Y02A	TECHNOLOGIES FOR ADAPTATION TO CLIMATE CHANGE
# Y02B	CLIMATE CHANGE MITIGATION TECHNOLOGIES RELATED TO BUILDINGS, e.g. HOUSING, HOUSE APPLIANCES OR RELATED END-USER APPLICATIONS
# Y02C	CAPTURE, STORAGE, SEQUESTRATION OR DISPOSAL OF GREENHOUSE GASES [GHG]
# Y02D	CLIMATE CHANGE MITIGATION TECHNOLOGIES IN INFORMATION AND COMMUNICATION TECHNOLOGIES [ICT], I.E. INFORMATION AND COMMUNICATION TECHNOLOGIES AIMING AT THE REDUCTION OF THEIR OWN ENERGY USE
# Y02E	REDUCTION OF GREENHOUSE GAS [GHG] EMISSIONS, RELATED TO ENERGY GENERATION, TRANSMISSION OR DISTRIBUTION
# Y02P	CLIMATE CHANGE MITIGATION TECHNOLOGIES IN THE PRODUCTION OR PROCESSING OF GOODS
# Y02T	CLIMATE CHANGE MITIGATION TECHNOLOGIES RELATED TO TRANSPORTATION
# Y02W	CLIMATE CHANGE MITIGATION TECHNOLOGIES RELATED TO WASTEWATER TREATMENT OR WASTE MANAGEMENT
# Y04S	SYSTEMS INTEGRATING TECHNOLOGIES RELATED TO POWER NETWORK OPERATION, COMMUNICATION OR INFORMATION TECHNOLOGIES FOR IMPROVING THE ELECTRICAL POWER GENERATION, TRANSMISSION, DISTRIBUTION, MANAGEMENT OR USAGE, i.e. SMART GRIDS
#


##### Read CSV files:

df_patent_All_CPC = pd.read_csv('Results/0_RawPatents.csv', header=0, dtype='unicode', low_memory=False)

##### I only need those ros with non-empty all_CPC column.

df_patent_All_CPC = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].notna()]


### All

# df_patent_All_CPC_all = df_patent_All_CPC[df_patent_All_CPC.apply(lambda r: r.str.contains('Y02|Y04S', case=False).any(), axis=1)]
df_patent_All_CPC_all = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02", regex=False) | df_patent_All_CPC['all_CPC'].str.contains("Y04S", regex=False)]

df_patent_All_CPC_all.to_csv('Results/Step1/1_df_patent_USPTO_Greens_all.csv', index=False)


# Y02A

df_patent_All_CPC_Y02A = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02A", regex=False)]

df_patent_All_CPC_Y02A.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02A.csv', index=False)

# Y02B

df_patent_All_CPC_Y02B = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02B", regex=False)]

df_patent_All_CPC_Y02B.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02B.csv', index=False)

# Y02C

df_patent_All_CPC_Y02C = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02C", regex=False)]

df_patent_All_CPC_Y02C.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02C.csv', index=False)

# Y02D

df_patent_All_CPC_Y02D = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02D", regex=False)]

df_patent_All_CPC_Y02D.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02D.csv', index=False)

# Y02E

df_patent_All_CPC_Y02E = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02E", regex=False)]

df_patent_All_CPC_Y02E.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02E.csv', index=False)

# Y02P

df_patent_All_CPC_Y02P = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02P", regex=False)]

df_patent_All_CPC_Y02P.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02P.csv', index=False)


# Y02T

df_patent_All_CPC_Y02T = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02T", regex=False)]

df_patent_All_CPC_Y02T.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02T.csv', index=False)

# Y02W

df_patent_All_CPC_Y02W = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y02W", regex=False)]

df_patent_All_CPC_Y02W.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y02W.csv', index=False)

# Y04S

df_patent_All_CPC_Y04S = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].str.contains("Y04S", regex=False)]

df_patent_All_CPC_Y04S.to_csv('Results/Step1/1_df_patent_USPTO_Greens_Y04S.csv', index=False)


# python3 2_GreenFeatureCandidates.py Y02A &&
# python3 2_GreenFeatureCandidates.py Y02B &&
# python3 2_GreenFeatureCandidates.py Y02C &&
# python3 2_GreenFeatureCandidates.py Y02D &&
# python3 2_GreenFeatureCandidates.py Y02E &&
# python3 2_GreenFeatureCandidates.py Y02P &&
# python3 2_GreenFeatureCandidates.py Y02T &&
# python3 2_GreenFeatureCandidates.py Y02W &&
# python3 2_GreenFeatureCandidates.py Y04S


