import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from CustomScaler import *

data = pd.read_csv('high_diamond_ranked_10min.csv')
'''

print("前五行數據:", data.head())
print("數據形狀:", data.shape)
pd.set_option('display.width', 10)  # 設置Console每一行展示的最大寬度，螢幕一行顯示滿之後才會進行換行
print("數據行名:", data.columns)
pd.set_option('display.width', 80)  #設置Console每一行展示的最大寬度，螢幕一行顯示滿之後才會進行換行
print("數據概覽:", data.describe())
print("數據概覽:", data.info())
'''

data.dropna(axis=0, how='any', inplace=True)  
data = data.drop(['gameId'], axis=1)  

plt.figure(figsize=(18, 15))
sns.heatmap(round(data.corr(), 1), cmap="coolwarm", annot=True, linewidths=.5)
plt.savefig('correlation matrix.jpg', bbox_inches='tight')


def remove_redundancy(r):
    to_remove = []
    for i in range(len(r.columns)):
        for j in range(i):
            if (abs(r.iloc[i, j]) >= 1 and (r.columns[j] not in to_remove)):
                print("相關性:", r.iloc[i, j], r.columns[j], r.columns[i])
                to_remove.append(r.columns[i])
    return to_remove


clean_data = data.drop(remove_redundancy(data.corr()), axis=1)

'''
pd.set_option('display.width', 10)
print("初步處理後的數據:", clean_data.columns)
'''

# 將擊殺小兵和野怪數量結合
clean_data['blueMinionsTotales'] = clean_data['blueTotalMinionsKilled'] + clean_data['blueTotalJungleMinionsKilled']
clean_data['redMinionsTotales'] = clean_data['redTotalMinionsKilled'] + clean_data['redTotalJungleMinionsKilled']
clean_data = clean_data.drop(['blueTotalMinionsKilled'], axis=1)
clean_data = clean_data.drop(['blueTotalJungleMinionsKilled'], axis=1)
clean_data = clean_data.drop(['redTotalMinionsKilled'], axis=1)
clean_data = clean_data.drop(['redTotalJungleMinionsKilled'], axis=1)

# 等級和經驗分析
plt.figure(figsize=(12, 12))
plt.subplot(121)
sns.scatterplot(x='blueAvgLevel', y='blueTotalExperience', hue='blueWins', data=clean_data)
plt.title('blue')
plt.xlabel('blueAvgLevel')
plt.ylabel('blueTotalExperience')
plt.grid(True)
plt.subplot(122)
sns.scatterplot(x='redAvgLevel', y='redTotalExperience', hue='blueWins', data=clean_data)
plt.title('red')
plt.xlabel('redAvgLevel')
plt.ylabel('redTotalExperience')
plt.grid(True)
plt.savefig('LevelxExperience.jpg', bbox_inches='tight')

clean_data = clean_data.drop(['blueAvgLevel'], axis=1)
clean_data = clean_data.drop(['redAvgLevel'], axis=1)

sns.set(font_scale=1.5)
plt.figure(figsize=(20, 20))
sns.set_style("whitegrid")

# 擊殺和被擊殺數繪製散佈圖
plt.subplot(321)
sns.scatterplot(x='blueKills', y='blueDeaths', hue='blueWins', data=clean_data)
plt.title('blueKills&&blueDeaths')
plt.xlabel('blueKills')
plt.ylabel('blueDeaths')
plt.grid(True)

# 助攻數繪製散佈圖
plt.subplot(322)
sns.scatterplot(x='blueAssists', y='redAssists', hue='blueWins', data=clean_data)
plt.title('Assists')
plt.xlabel('blueAssists')
plt.ylabel('redAssists')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 雙方金幣數繪製散佈圖
plt.subplot(323)
sns.scatterplot(x='blueTotalGold', y='redTotalGold', hue='blueWins', data=clean_data)
plt.title('TotalGold')
plt.xlabel('blueTotalGold')
plt.ylabel('redTotalGold')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 雙方經驗繪製散佈圖
plt.subplot(324)
sns.scatterplot(x='blueTotalExperience', y='redTotalExperience', hue='blueWins', data=clean_data)
plt.title('Experience')
plt.xlabel('blueTotalExperience')
plt.ylabel('redTotalExperience')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 雙方插眼數量繪製散佈圖
plt.subplot(325)
sns.scatterplot(x='blueWardsPlaced', y='redWardsPlaced', hue='blueWins', data=clean_data)
plt.title('WardsPlaced')
plt.xlabel('blueWardsPlaced')
plt.ylabel('redWardsPlaced')
plt.tight_layout(pad=1.5)
plt.grid(True)

# 擊殺的小兵和野怪數量繪製散佈圖
plt.subplot(326)
sns.scatterplot(x='blueMinionsTotales', y='redMinionsTotales', hue='blueWins', data=clean_data)
plt.title('MinionsTotales')
plt.xlabel('blueMinionsTotales')
plt.ylabel('redMinionsTotales')
plt.tight_layout(pad=1.5)
plt.grid(True)
plt.savefig('數據分析.jpg', bbox_inches='tight')

clean_data['WardsPlacedDiff'] = clean_data['blueWardsPlaced'] - clean_data['redWardsPlaced']
clean_data['WardsDestroyedDiff'] = clean_data['blueWardsDestroyed'] - clean_data['redWardsDestroyed']
clean_data['AssistsDiff'] = clean_data['blueAssists'] - clean_data['redAssists']
clean_data['blueHeraldsDiff'] = clean_data['blueHeralds'] - clean_data['redHeralds']
clean_data['blueDragonsDiff'] = clean_data['blueDragons'] - clean_data['redDragons']
clean_data['blueTowersDestroyedDiff'] = clean_data['blueTowersDestroyed'] - clean_data['redTowersDestroyed']
clean_data['EliteMonstersDiff'] = clean_data['blueEliteMonsters'] - clean_data['redEliteMonsters']
clean_data = clean_data.drop(['blueWardsPlaced'], axis=1)
clean_data = clean_data.drop(['redWardsPlaced'], axis=1)
clean_data = clean_data.drop(['blueWardsDestroyed'], axis=1)
clean_data = clean_data.drop(['redWardsDestroyed'], axis=1)
clean_data = clean_data.drop(['blueAssists'], axis=1)
clean_data = clean_data.drop(['redAssists'], axis=1)
clean_data = clean_data.drop(['blueHeralds'], axis=1)
clean_data = clean_data.drop(['redHeralds'], axis=1)
clean_data = clean_data.drop(['blueTowersDestroyed'], axis=1)
clean_data = clean_data.drop(['redTowersDestroyed'], axis=1)
clean_data = clean_data.drop(['blueDragons'], axis=1)
clean_data = clean_data.drop(['redDragons'], axis=1)
clean_data = clean_data.drop(['blueEliteMonsters'], axis=1)
clean_data = clean_data.drop(['redEliteMonsters'], axis=1)
clean_data = clean_data.drop(['redTotalGold'], axis=1)  
clean_data = clean_data.drop(['redTotalExperience'], axis=1)


sns.catplot(x="blueWins", y="blueGoldDiff", hue="blueFirstBlood", data=clean_data)
plt.savefig('FirstBlood.jpg', bbox_inches='tight')
sns.catplot(x="blueWins", y="blueGoldDiff", hue="blueDragonsDiff", data=clean_data)
plt.savefig('Dragon.jpg', bbox_inches='tight')
sns.catplot(x="blueWins", y="blueGoldDiff", hue="EliteMonstersDiff", data=clean_data)
plt.savefig('EliteMonsters.jpg', bbox_inches='tight')

print("最終數據:", clean_data.columns)


unscaled_inputs = clean_data.filter([
    'blueFirstBlood',
    'blueKills',
    'blueDeaths',
    'blueTotalGold',
    'blueTotalExperience',
    'blueGoldDiff',
    'blueExperienceDiff',
    'blueMinionsTotales',
    'redMinionsTotales',
    'WardsPlacedDiff',
    'WardsDestroyedDiff',
    'AssistsDiff',
    'blueHeraldsDiff',
    'blueDragonsDiff',
    'blueTowersDestroyedDiff',
    'EliteMonstersDiff'], axis=1)
target = clean_data.filter(['blueWins'])





# 數據縮放要忽略的類
columns_to_omit = ['blueFirstBlood', 'blueDragonsDiff']  #忽略FirstBlood，因為他是分類變量

# 根據要縮放的col創建列表
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
blue_scaler = CustomScaler(columns_to_scale)
blue_scaler.fit(unscaled_inputs)
scaled_inputs = blue_scaler.transform(unscaled_inputs)
pd.set_option('display.width', 80)
print("標準化處理後的數據:", scaled_inputs)

# 數據切片
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=2)
print("訓練數據:", x_train.shape, y_train.shape, "測試數據:", x_test.shape, y_test.shape)

# 模型訓練
reg = LogisticRegression()
reg.fit(x_train, y_train)


variables = unscaled_inputs.columns.values
intercept = reg.intercept_  
summary_table = pd.DataFrame(columns=['Variables'], data=variables)
summary_table['Coef'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table['Odds Ratio'] = np.exp(summary_table.Coef)
summary_table.sort_values(by=['Odds Ratio'], ascending=False)
print("模型變量評價:", summary_table.sort_values(by=['Odds Ratio'], ascending=False))

# 模型測試
print("訓練數據評分:", reg.score(x_train, y_train))
print("測試數據評分:", reg.score(x_test, y_test))

predicted_prob = reg.predict_proba(x_test)
data['predicted'] = reg.predict_proba(scaled_inputs)[:, 1]
print("經過預測後的包含預測結果的完整數據集:", data)

# 原始數據和勝率分析對比
col_n = ['blueWins', 'predicted']
a = pd.DataFrame(data, columns=col_n)
print("原始數據和勝率分析對比:", a)
