import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from os import listdir
from os.path import isfile, join

def get_num_res(result):
    outcome_dict = {'W':1, 'D':0, 'L':-1}
    return outcome_dict[result]

def get_num_team(team):
    team_dict = {
        'Arsenal':0, 'Aston Villa':1, 'Brentford':2, 'Brighton':3, 'Burnley':4,
       'Chelsea':5, 'Crystal Palace':6, 'Everton':7, 'Leeds United':8,
       'Leicester City':9, 'Liverpool':10, 'Manchester City':11, 'Manchester Utd':12,
       'Newcastle Utd':13, 'Norwich City':14, 'Southampton':15, 'Tottenham':16,
       'Watford':17, 'West Ham':18, 'Wolves':19
    }
    
    return team_dict[team]

def preprocess(data_dir):
    df_lst = []
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    leag_tbl_cols = ['MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'xG', 'xGA', 'xGD', 'xGD/90', 0, 1, 2, 3, 4]
    

    print('Reading team data files')
    files.remove('.DS_Store')
    for f in files:
        print(f)
        if f != 'league_table_overall.csv':
            temp = pd.read_csv(data_dir+f, skiprows=1)
            temp = temp.drop('Squad',axis=1)
            
        else:
            temp = pd.read_csv(data_dir+f)
            foo = lambda x: pd.Series([i for i in reversed(x.split(' '))])
            rev = temp['Last 5'].apply(foo)
            
            rev[0] = rev[0].apply(get_num_res)
            rev[1] = rev[1].apply(get_num_res)
            rev[2] = rev[2].apply(get_num_res)
            rev[3] = rev[3].apply(get_num_res)
            rev[4] = rev[4].apply(get_num_res)
            
            temp = pd.concat([temp, rev],axis=1)
            temp = temp[leag_tbl_cols]
        
        df_lst.append(temp)
            
            
        
    team_data = pd.concat(df_lst,axis=1)
    scaler = MinMaxScaler()
    scaler.fit(team_data.values)
    team_data_scaled = scaler.transform(team_data.values)
    
    #print(team_data_scaled)
    team_data = pd.DataFrame(team_data_scaled, columns =list(team_data.columns))
    team_data = team_data.fillna(0.19)
    
    pickle.dump(list(team_data.columns), open( 'train_team_data_cols.pkl', 'wb'))
    #print(list(team_data.columns))
    team_data['squad'] = pd.read_csv(data_dir+'pass_type.csv', skiprows=1)['Squad'].apply(get_num_team)
    #print(team_data)
    pickle.dump(scaler, open( 'train_minmax_scaler.pkl', 'wb'))
    pickle.dump(team_data, open( 'train_team_data_df.pkl', 'wb'))
    
    
    
    


if __name__ == '__main__':
    data_dir = '/Users/pamelakatali/Downloads/soccer_project/data/'
    preprocess(data_dir)
    