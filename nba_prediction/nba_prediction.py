# -*- coding: utf-8 -*-

# step1 import packages
import csv
import math
import random

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# step2 设置回归训练时所需用到的参数变量：
# 当每支队伍没有elo等级分时，赋予其基础elo等级分
BASE_ELO = 1600
TEAM_ELO_MAP = {}
TEAM_STAT_MAP = {}
DATA_FOLDER = 'data'  # 存放数据的目录

X = []  # 特征矩阵
y = []  # 结果


# step3 在最开始需要初始化数据，从t、o和m表格中读入数据，去除一些无关数据并将这三个表格通过team属性列进行连接
def initialize_data(m_stat, o_stat, t_stat):
    """
    根据每支队伍的miscellaneous opponent，team统计数据csv文件进行初始化
    :param m_stat: 
    :param o_stat: 
    :param t_stat: 
    :return: 
    """
    new_m_stat = m_stat.drop(['Rk', 'Arena'], axis=1)
    new_o_stat = o_stat.drop(['Rk', 'G', 'MP'], axis=1)
    new_t_stat = t_stat.drop(['Rk', 'G', 'MP'], axis=1)

    df_team_stats = pd.merge(new_m_stat, new_o_stat, how='left', on='Team')
    df_team_stats = pd.merge(df_team_stats, new_t_stat, how='left', on='Team')
    return df_team_stats.set_index('Team', inplace=False, drop=True)


def get_elo(team):
    """
    获取每支队伍的elo score等级分函数，当在开始没有等级分时，将其赋予初始base_elo值
    :param team: 
    :return: 
    """
    try:
        return TEAM_ELO_MAP[team]
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        TEAM_ELO_MAP[team] = BASE_ELO
        return TEAM_ELO_MAP[team]


def calculate_elo(win_team, lose_team):
    """
    计算每个球队的elo值
    :param win_team: 
    :param lose_team: 
    :return: 
    """
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # 根据rank级别修改k值
    if winner_rank < 2100:
        k = 32
    elif 2100 <= winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


def build_data_set(all_data):
    """
    基于我们初始好的统计数据，及每支队伍的elo score计算结果，建立对应2015~2016年常规赛和季后赛中每场比赛的数据集（在主客场比赛时，我们认为主场作战的队伍更加有优势一点，因此会给主场作战队伍相应加上100等级分）
    :param all_data: 
    :return: 
    """
    print("building data set..")
    x = []
    skip = 0
    for index, row in all_data.iterrows():

        w_team = row['WTeam']
        l_team = row['LTeam']

        # 获取最初的elo或是每个队伍最初的elo值
        team1_elo = get_elo(w_team)
        team2_elo = get_elo(l_team)

        # 给主场比赛的队伍加上100的elo值
        if row['WLoc'] == 'h':
            team1_elo += 100
        else:
            team2_elo += 100

        # 把elo当为评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # 添加我们从basketball reference.com获得的每个队伍的统计信息
        for key, value in TEAM_STAT_MAP.loc[w_team].iteritems():
            team1_features.append(value)
        for key, value in TEAM_STAT_MAP.loc[l_team].iteritems():
            team2_features.append(value)

        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            x.append(team1_features + team2_features)
            y.append(0)
        else:
            x.append(team2_features + team1_features)
            y.append(1)

        if skip == 0:
            print(x)
            skip = 1

        # 根据这场比赛的数据更新队伍的elo值
        new_winner_rank, new_loser_rank = calculate_elo(w_team, l_team)
        TEAM_ELO_MAP[w_team] = new_winner_rank
        TEAM_ELO_MAP[l_team] = new_loser_rank

    return np.nan_to_num(x), y


def predict_winner(team_1, team_2, model):
    """
    利用模型对一场新的比赛进行胜负判断，并返回其胜利的概率
    :param team_1: 
    :param team_2: 
    :param model: 
    :return: 
    """
    features = []

    # team 1，客场队伍
    features.append(get_elo(team_1))
    for key, value in TEAM_STAT_MAP.loc[team_1].iteritems():
        features.append(value)

    # team 2，主场队伍
    features.append(get_elo(team_2) + 100)
    for key, value in TEAM_STAT_MAP.loc[team_2].iteritems():
        features.append(value)

    features = np.nan_to_num(features)
    return model.predict_proba([features])


if __name__ == '__main__':
    m_stat = pd.read_csv(DATA_FOLDER + '/15-16_Miscellaneous_Stat.csv')
    o_stat = pd.read_csv(DATA_FOLDER + '/15-16_Opponent_Per_Game_Stat.csv')
    t_stat = pd.read_csv(DATA_FOLDER + '/15-16_Team_Per_Game_Stat.csv')

    TEAM_STAT_MAP = initialize_data(m_stat, o_stat, t_stat)

    result_data = pd.read_csv(DATA_FOLDER + '/15-16_Result.csv')
    X, y = build_data_set(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    # 使用sklearn的logistic regression方法建立回归模型
    model = linear_model.LogisticRegression()
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    print(cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1).mean())

    # 利用训练好的model在16-17年的比赛中进行预测
    print('Predicting on new schedule..')
    schedule1617 = pd.read_csv(DATA_FOLDER + '/16-17_Schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
        team1 = row['Vteam']
        team2 = row['Hteam']
        predictions = predict_winner(team1, team2, model)
        prob = predictions[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])

    with open('16-17_Predicted_Result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)
