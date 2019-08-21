import torch
from torch import nn

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')
sys.setrecursionlimit(500000)
import feat as fe

EPOCH_NUM = 300

class NN(nn.Module):
    """Neural Network model"""
    def __init__(self, input_size, output_size=1):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, output_size)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def hotting(data, k=35):
    data_shape = data.shape
    if len(data_shape) ==1:
        tmp = [np.eye(k)[data.astype(int)[0]],np.eye(k)[data.astype(int)[1]]]
        res = np.concatenate(tmp ,axis=0)
        return res
    tmp = [np.eye(k)[data[:,0].astype(int)], np.eye(k)[data[:,1].astype(int)]]
    return np.concatenate(tmp ,axis=1)


def prepare_data():
    """read data from new.xlsx
       return: data(pd.dataframe)
    """
    # prepare features
    data = pd.read_excel('./new.xlsx', sheet_name=1, usecols='C, E, I:M, O:AA, CJ, CL, CP:CT, CV:DH', header=0)
    data = data.dropna()

    return data


def prepare_season_data():
    """read new.xlsx and extract season data, write to season_data.xlsx"""
    season_data = pd.read_excel('./new.xlsx', sheet_name=1, usecols='C, E, I:M, O:AA, CJ, CL, CP:CT, CV:DH', header=0)
    season_data = season_data.dropna()
    offseason_data = pd.read_excel('./new.xlsx', sheet_name=2, usecols='C, E, I:M, O:AA, CJ, CL, CP:CT, CV:DH', header=0)
    offseason_data = offseason_data.dropna()

    return season_data, offseason_data


def prepare_win_lose_data(data):
    """input: data(pd.dataframe)
       return: team(pd.dataframe)
               labels(pd.dataframe)
    """
    team = data.iloc[:, [1, 21]]

    # prepare labels
    labels = data.iloc[:, 2]
    for idx, label in labels.iteritems():
        if label == '胜':
            labels[idx] = 1
        elif label == '负':
            labels[idx] = 0
        else:
            assert False
    return team, labels


def prepare_feature_data(data):
    """input: data(pd.dataframe)
       return: feature(pd.dataframe)
               labels(pd.dataframe)
    """
    team = data.iloc[:, [1, 21]]

    # prepare features
    features = data.drop(data.columns[[0, 1, 2, 21, 22]], axis=1)
    return team, features


def encodeteam(team):
    """encode team name to number"""
    encode = {
        '底特律活塞': 0,
        '密尔沃基雄鹿': 1,
        '金州勇士': 2,
        '洛杉矶快船': 3,
        '波士顿凯尔特人': 4, 
        '印第安纳步行者': 5, 
        '休斯顿火箭': 6,
        '犹他爵士': 7,
        '奥兰多魔术': 8,
        '多伦多猛龙': 9,
        '丹佛掘金': 10,
        '圣安东尼奥马刺': 11,
        '布鲁克林篮网': 12,
        '费城76人': 13, 
        '俄克拉荷马雷霆': 14,
        '波特兰开拓者': 15,
        '华盛顿奇才': 16, 
        '克里夫兰骑士': 17,
        '迈阿密热火': 18,
        '明尼苏达森林狼': 19,
        '新奥尔良鹈鹕': 20,
        '芝加哥公牛': 21,
        '亚特兰大老鹰': 22,
        '孟菲斯灰熊': 23,
        '夏洛特黄蜂': 24,
        '达拉斯小牛': 25,
        '夏洛特山猫': 26,
        '纽约尼克斯': 27,
        '洛杉矶湖人': 28,
        '菲尼克斯太阳': 29,
        '新泽西篮网': 30,
        '萨克拉门托国王': 31,
        '西雅图超音速': 32,
        '华盛顿子弹': 33,
        '对手': 34
    }
    team = team.replace(encode)
    return team


def split_data(data):
    """split training set and testing set.
       Write to xlsx file
    """
    idxs = np.random.randint(0, len(data), size=int(0.2*len(data)))
    train_data = data.drop(data.index[idxs])
    train_data.to_excel('./train_offseason_data.xlsx')
    test_data = data.iloc[idxs]
    test_data.to_excel('./test_offseason_data.xlsx')


def load_train_test():
    """load data from train_data.xlsx and test_data.xlsx"""
    train_season_data   = pd.read_excel('./train_season_data.xlsx')
    test_season_data    = pd.read_excel('./test_season_data.xlsx')
    train_offseason_data   = pd.read_excel('./train_offseason_data.xlsx')
    test_offseason_data    = pd.read_excel('./test_offseason_data.xlsx')

    return train_season_data, test_season_data, train_offseason_data, test_offseason_data


def train_win_lose(net, train_season_data, train_season_labels, train_offseason_data, train_offseason_labels):
    """training
       train_data: tensor
       train_labels: tensor
    """
    print('Start training...')
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    batchSize = 20
    import random
    for epoch in range(EPOCH_NUM):
        batchList = []
        for inputs, target in zip(train_season_data, train_season_labels):
            batchList.append([inputs,target])
        for inputs, target in zip(train_offseason_data, train_offseason_labels):
            batchList.append([inputs,target])
        random.shuffle(batchList)
        for i in range(len(batchList)//batchSize):
            batchData = batchList[batchSize*i: batchSize*(i+1)]
            inputs = torch.stack([x[0] for x in batchData],dim=0)
            target = torch.tensor([[x[1]] for x in batchData])
            # print(inputs.size())
            # target = torch.tensor([target])
            out = net(inputs)
            loss = criterion(out, target) * 0.8

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break

        # for inputs, target in zip(train_offseason_data, train_offseason_labels):
        #     target = torch.tensor([target])
            # print(inputs.size())
            # out = net(inputs)
            # loss = criterion(out, target)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # break

        if (epoch + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(epoch + 1, loss.item()))
    return net


def test_win_lose(net, test_data, test_labels, probList=[], alpha=0.5, show=True):
    """Test
       test_data: tensor
       test_labels: tensor
    """
    if show:
        print('Testing...')
    running_corrects = 0
    positive = 0
    negative = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():
        for i,(inputs, target) in enumerate(zip(test_data, test_labels)):
            out = net(inputs)
            prob = torch.sigmoid(out)
            if len(probList)>0:
                score = (1-alpha)*prob.item() + alpha*probList[i] 
            else:
                score = prob.item()
            if score > 0.5:
                pred = 1
            else:
                pred = 0
            running_corrects += (pred == target.item())
            if target.item() == 1:
                positive += 1
                if pred == 1:
                    TP += 1
                else:
                    FP += 1
            elif target.item() == 0:
                negative += 1
                if pred == 1:
                    FN += 1
                else:
                    TN += 1
            else:
                assert False
    acc = running_corrects / len(test_labels)
    if show:
        print('Accuracy:', acc)
        print('Positive sample:', positive)
        print('Negative sample:', negative)
        print('True Positive:', TP)
        print('True Negative:', TN)
        print('False Positive:', FP)
        print('False Negative:', FN)
        print('Precision:', TP/(TP+FP))
        print('Recall:', TP/(TP+FN))
        print('F1 Score:', 2*TP/(2*TP + FP + FN))
    return acc


def train_feature_net(net, train_season_data, train_season_labels, train_offseason_data, train_offseason_labels):
    """training
       train_data: tensor
       train_labels: tensor
    """
    print('Start feature training...')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(100):
        for inputs, target in zip(train_season_data, train_season_labels):
            out = net(inputs)
            loss = criterion(out, target) * 0.8

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        for inputs, target in zip(train_offseason_data, train_offseason_labels):
            out = net(inputs)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        if (epoch + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(epoch + 1, loss.item()))
        break
    return net
def test_feature_net_ora(net, team1, team2):
    """Test
       test_data: tensor
       test_labels: tensor
    """
    encode = {
        '底特律活塞': 0,
        '密尔沃基雄鹿': 10,
        '金州勇士': 20,
        '洛杉矶快船': 30,
        '波士顿凯尔特人': 40,
        '印第安纳步行者': 50,
        '休斯顿火箭': 60,
        '犹他爵士': 70,
        '奥兰多魔术': 80,
        '多伦多猛龙': 90,
        '丹佛掘金': 100,
        '圣安东尼奥马刺': 110,
        '布鲁克林篮网': 120,
        '费城76人': 130,
        '俄克拉荷马雷霆': 140,
        '波特兰开拓者': 150,
        '华盛顿奇才': 160,
        '克里夫兰骑士': 170,
        '迈阿密热火': 180,
        '明尼苏达森林狼': 190,
        '新奥尔良鹈鹕': 200,
        '芝加哥公牛': 210,
        '亚特兰大老鹰': 220,
        '孟菲斯灰熊': 230,
        '夏洛特黄蜂': 240,
        '达拉斯小牛': 250,
        '夏洛特山猫': 260,
        '纽约尼克斯': 270,
        '洛杉矶湖人': 280,
        '菲尼克斯太阳': 290,
        '新泽西篮网': 300,
        '萨克拉门托国王': 310,
        '西雅图超音速': 320,
        '华盛顿子弹': 330,
        '对手': 210
    }
    inputs =  np.array([encode[team1], encode[team2]]).astype(np.float)
    inputs = torch.from_numpy(inputs)
    net = net.double()
    with torch.no_grad():
        output = net(inputs)
    print('{}: 篮板:{}, 进攻篮板:{}, 防守篮板:{}, 抢断:{}, 盖帽:{}, 得分:{}, 出手:{}, 命中:{}, %:{}, \
          罚球出手:{}, 罚球命中:{}, 罚球%:{}, 三分出手:{}, 三分命中:{}, 三分%:{}, 助攻:{}, 失误:{}, \
          犯规:{}'.format(team1, output[0].item(), output[1].item(), output[2].item(), 
                    output[3].item(), output[4].item(), output[5].item(), output[6].item(), 
                    output[7].item(), output[8].item(), output[9].item(), output[10].item(), 
                    output[11].item(), output[12].item(), output[13].item(), output[14].item(), 
                    output[15].item(), output[16].item(), output[17].item()))
    print('{}: 篮板:{}, 进攻篮板:{}, 防守篮板:{}, 抢断:{}, 盖帽:{}, 得分:{}, 出手:{}, 命中:{}, %:{}, \
          罚球出手:{}, 罚球命中:{}, 罚球%:{}, 三分出手:{}, 三分命中:{}, 三分%:{}, 助攻:{}, 失误:{}, \
          犯规:{}'.format(team2, output[18].item(), output[19].item(), output[20].item(), 
                        output[21].item(), output[22].item(), output[23].item(), 
                        output[24].item(), output[25].item(), output[26].item(), 
                        output[27].item(), output[28].item(), output[29].item(), 
                        output[30].item(), output[31].item(), output[32].item(), 
                        output[33].item(), output[34].item(), output[35].item()))
    return team1, team2, output

def test_feature_net(net, team1, team2):
    """Test
       test_data: tensor
       test_labels: tensor
    """
    encode = {
        '底特律活塞': 0,
        '密尔沃基雄鹿': 1,
        '金州勇士': 2,
        '洛杉矶快船': 3,
        '波士顿凯尔特人': 4, 
        '印第安纳步行者': 5, 
        '休斯顿火箭': 6,
        '犹他爵士': 7,
        '奥兰多魔术': 8,
        '多伦多猛龙': 9,
        '丹佛掘金': 10,
        '圣安东尼奥马刺': 11,
        '布鲁克林篮网': 12,
        '费城76人': 13, 
        '俄克拉荷马雷霆': 14,
        '波特兰开拓者': 15,
        '华盛顿奇才': 16, 
        '克里夫兰骑士': 17,
        '迈阿密热火': 18,
        '明尼苏达森林狼': 19,
        '新奥尔良鹈鹕': 20,
        '芝加哥公牛': 21,
        '亚特兰大老鹰': 22,
        '孟菲斯灰熊': 23,
        '夏洛特黄蜂': 24,
        '达拉斯小牛': 25,
        '夏洛特山猫': 26,
        '纽约尼克斯': 27,
        '洛杉矶湖人': 28,
        '菲尼克斯太阳': 29,
        '新泽西篮网': 30,
        '萨克拉门托国王': 31,
        '西雅图超音速': 32,
        '华盛顿子弹': 33,
        '对手': 34
    }
    inputs =  np.array([encode[team1], encode[team2]]).astype(np.float)
    inputs = torch.from_numpy(inputs)
    net = net.double()
    with torch.no_grad():
        output = net(inputs)
    print('{}: 篮板:{}, 进攻篮板:{}, 防守篮板:{}, 抢断:{}, 盖帽:{}, 得分:{}, 出手:{}, 命中:{}, %:{}, \
          罚球出手:{}, 罚球命中:{}, 罚球%:{}, 三分出手:{}, 三分命中:{}, 三分%:{}, 助攻:{}, 失误:{}, \
          犯规:{}'.format(team1, output[0].item(), output[1].item(), output[2].item(), 
                    output[3].item(), output[4].item(), output[5].item(), output[6].item(), 
                    output[7].item(), output[8].item(), output[9].item(), output[10].item(), 
                    output[11].item(), output[12].item(), output[13].item(), output[14].item(), 
                    output[15].item(), output[16].item(), output[17].item()))
    print('{}: 篮板:{}, 进攻篮板:{}, 防守篮板:{}, 抢断:{}, 盖帽:{}, 得分:{}, 出手:{}, 命中:{}, %:{}, \
          罚球出手:{}, 罚球命中:{}, 罚球%:{}, 三分出手:{}, 三分命中:{}, 三分%:{}, 助攻:{}, 失误:{}, \
          犯规:{}'.format(team2, output[18].item(), output[19].item(), output[20].item(), 
                        output[21].item(), output[22].item(), output[23].item(), 
                        output[24].item(), output[25].item(), output[26].item(), 
                        output[27].item(), output[28].item(), output[29].item(), 
                        output[30].item(), output[31].item(), output[32].item(), 
                        output[33].item(), output[34].item(), output[35].item()))
    return team1, team2, output


def compare_feature(team1, team2, output):
    """compare the predicted features"""
    if output[0].item() > output[18].item():
        print('篮板: {}胜'.format(team1))
    else:
        print('篮板: {}胜'.format(team2))

    if output[3].item() > output[21].item():
        print('抢断: {}胜'.format(team1))
    else:
        print('抢断: {}胜'.format(team2))

    if output[4].item() > output[22].item():
        print('盖帽: {}胜'.format(team1))
    else:
        print('盖帽: {}胜'.format(team2))


def predict(team1, team2, win_lose_net, feature_net):
    encode = {
        '底特律活塞': 0,
        '密尔沃基雄鹿': 1,
        '金州勇士': 2,
        '洛杉矶快船': 3,
        '波士顿凯尔特人': 4, 
        '印第安纳步行者': 5, 
        '休斯顿火箭': 6,
        '犹他爵士': 7,
        '奥兰多魔术': 8,
        '多伦多猛龙': 9,
        '丹佛掘金': 10,
        '圣安东尼奥马刺': 11,
        '布鲁克林篮网': 12,
        '费城76人': 13, 
        '俄克拉荷马雷霆': 14,
        '波特兰开拓者': 15,
        '华盛顿奇才': 16, 
        '克里夫兰骑士': 17,
        '迈阿密热火': 18,
        '明尼苏达森林狼': 19,
        '新奥尔良鹈鹕': 20,
        '芝加哥公牛': 21,
        '亚特兰大老鹰': 22,
        '孟菲斯灰熊': 23,
        '夏洛特黄蜂': 24,
        '达拉斯小牛': 25,
        '夏洛特山猫': 26,
        '纽约尼克斯': 27,
        '洛杉矶湖人': 28,
        '菲尼克斯太阳': 29,
        '新泽西篮网': 30,
        '萨克拉门托国王': 31,
        '西雅图超音速': 32,
        '华盛顿子弹': 33,
        '对手': 34
    }

    inputs =  np.array([encode[team1], encode[team2]]).astype(np.float)
    inputs = hotting(inputs)
    inputs = torch.from_numpy(inputs)
    win_lose_net = win_lose_net.double()
    out = win_lose_net(inputs)
    prob = torch.sigmoid(out)
    if prob.item() > 0.5:
        print('{}胜, 置信度:{}'.format(team1, (prob.item()-0.4)))
    else:
        print('{}胜, 置信度:{}'.format(team2, (0.6-prob.item())))

    # team1, team2, output = test_feature_net(feature_net, team1, team2)
    # compare_feature(team1, team2, output)


def predict_feat(team1, team2, feature_net):
    encode = {
        '底特律活塞': 0,
        '密尔沃基雄鹿': 10,
        '金州勇士': 20,
        '洛杉矶快船': 30,
        '波士顿凯尔特人': 40,
        '印第安纳步行者': 50,
        '休斯顿火箭': 60,
        '犹他爵士': 70,
        '奥兰多魔术': 80,
        '多伦多猛龙': 90,
        '丹佛掘金': 100,
        '圣安东尼奥马刺': 110,
        '布鲁克林篮网': 120,
        '费城76人': 130,
        '俄克拉荷马雷霆': 140,
        '波特兰开拓者': 150,
        '华盛顿奇才': 160,
        '克里夫兰骑士': 170,
        '迈阿密热火': 180,
        '明尼苏达森林狼': 190,
        '新奥尔良鹈鹕': 200,
        '芝加哥公牛': 210,
        '亚特兰大老鹰': 220,
        '孟菲斯灰熊': 230,
        '夏洛特黄蜂': 240,
        '达拉斯小牛': 250,
        '夏洛特山猫': 260,
        '纽约尼克斯': 270,
        '洛杉矶湖人': 280,
        '菲尼克斯太阳': 290,
        '新泽西篮网': 300,
        '萨克拉门托国王': 310,
        '西雅图超音速': 320,
        '华盛顿子弹': 330,
        '对手': 210
    }

    inputs =  np.array([encode[team1], encode[team2]]).astype(np.float)
    inputs = torch.from_numpy(inputs)

    # win_lose_net = win_lose_net.double()
    # out = win_lose_net(inputs)
    # prob = torch.sigmoid(out)
    # if prob.item() > 0.5:
    #     print('{}胜, 置信度:{}'.format(team1, (prob.item()-0.5)))
    # else:
    #     print('{}胜, 置信度:{}'.format(team2, (0.5-prob.item())))

    team1, team2, output = test_feature_net_ora(feature_net, team1, team2)
    compare_feature(team1, team2, output)

if __name__ == '__main__':
    train_season_data, test_season_data, train_offseason_data, test_offseason_data = load_train_test()

    """season data"""
    # train
    train_season_teams, train_season_win = prepare_win_lose_data(train_season_data)
    _, train_season_feature = prepare_feature_data(train_season_data)
    train_season_teams = encodeteam(train_season_teams)

    train_season_teams = train_season_teams.to_numpy().astype(np.float)
    train_season_teams = hotting(train_season_teams)

    train_season_win = train_season_win.values.astype(np.float)
    train_season_feature = train_season_feature.to_numpy().astype(np.float)

    train_season_teams = torch.from_numpy(train_season_teams).float()
    train_season_win = torch.from_numpy(train_season_win).float()
    train_season_feature = torch.from_numpy(train_season_feature).float()


    # test
    test_season_teams, test_season_win = prepare_win_lose_data(test_season_data)
    _, test_season_feature = prepare_feature_data(test_season_data)
    test_season_teams = encodeteam(test_season_teams)

    test_season_teams = test_season_teams.to_numpy().astype(np.float)
    test_season_teams = hotting(test_season_teams)
    test_season_win = test_season_win.values.astype(np.float)
    test_season_feature = test_season_feature.to_numpy().astype(np.float)

    test_season_teams = torch.from_numpy(test_season_teams).float()
    test_season_win = torch.from_numpy(test_season_win).float()
    test_season_feature = torch.from_numpy(test_season_feature).float()


    """off season data"""
    # train
    train_offseason_teams, train_offseason_win = prepare_win_lose_data(train_offseason_data)
    _, train_offseason_feature = prepare_feature_data(train_offseason_data)
    train_offseason_teams = encodeteam(train_offseason_teams)

    train_offseason_teams = train_offseason_teams.to_numpy().astype(np.float)
    train_offseason_teams = hotting(train_offseason_teams)
    train_offseason_win = train_offseason_win.values.astype(np.float)
    train_offseason_feature = train_offseason_feature.to_numpy().astype(np.float)

    train_offseason_teams = torch.from_numpy(train_offseason_teams).float()
    train_offseason_win = torch.from_numpy(train_offseason_win).float()
    train_offseason_feature = torch.from_numpy(train_offseason_feature).float()


    test_offseason_teams, test_offseason_win = prepare_win_lose_data(test_offseason_data)
    _, test_offseason_feature = prepare_feature_data(test_offseason_data)
    test_offseason_teams = encodeteam(test_offseason_teams)

    test_offseason_teams = test_offseason_teams.to_numpy().astype(np.float)
    test_offseason_teams = hotting(test_offseason_teams)
    test_offseason_win = test_offseason_win.values.astype(np.float)
    test_offseason_feature = test_offseason_feature.to_numpy().astype(np.float)

    test_offseason_teams = torch.from_numpy(test_offseason_teams).float()
    test_offseason_win = torch.from_numpy(test_offseason_win).float()
    test_offseason_feature = torch.from_numpy(test_offseason_feature).float()

    """For training"""
    if sys.argv[1]=='train':
        win_lose_net = NN(input_size=70, output_size=1)
        # win_lose_net = train_win_lose(win_lose_net, train_season_teams, train_season_win, \
        #                                 train_offseason_teams, train_offseason_win)
        season_acc = test_win_lose(win_lose_net, test_season_teams, test_season_win, show=False)
        print("team acc >>>", season_acc)
        # offseason_acc = test_win_lose(win_lose_net, test_offseason_teams, test_offseason_win, probList, 0.2)
        # print('offseason_acc combined >>> : ', offseason_acc)
        # test_season_feature = test_season_feature.chunk(2,1)[0]
        acc, probList = fe.test_by_feature(test_offseason_feature.chunk(2,1)[0], test_offseason_win)
        print('offseason feature acc >>> ',acc)

        maxacc = 0
        for alpha in np.arange(0.1,1.0,0.001):
            offseason_acc = test_win_lose(win_lose_net, test_offseason_teams, test_offseason_win, probList, alpha, show=False)
            # print('offseason_acc combined >>> : ',alpha, offseason_acc)
            maxacc = max(offseason_acc, maxacc)
        print('max combined offseason_acc >>>', maxacc)
        torch.save(win_lose_net, './model-acc{:.2f}.pkl'.format(offseason_acc))

        feature_net = NN(input_size=70, output_size=36)
        feature_net = train_feature_net(feature_net, train_season_teams, train_season_feature, \
                                            train_offseason_teams, train_offseason_feature)
        # torch.save(feature_net, './feature-model.pkl')

        # net = torch.load('./model-acc0.83-v1.pkl')
        if not os.path.exists('./model-acc0.83-v1.pkl'):
            # fe.train_win_lose(net, train_season_feature.chunk(2,1)[0], train_season_win, train_offseason_feature.chunk(2,1)[0], train_offseason_win)
            fe.test(net, test_offseason_feature.chunk(2,1)[0], test_offseason_win)
            torch.save(net, './model-acc0.83-v1.pkl')



    """For testing"""
    if sys.argv[1]=='test':
        team1 = input('请输入队伍1： ')
        team2 = input('请输入队伍2： ')
        win_lose_net = torch.load('./model-acc0.56.pkl')
        # season_acc = test_win_lose(win_lose_net, test_season_teams, test_season_win)
        # offseason_acc = test_win_lose(win_lose_net, test_offseason_teams, test_offseason_win)
        # acc = (season_acc+offseason_acc)/2

        feature_net = torch.load('./feature-model.pkl')
        predict(team1, team2, win_lose_net, feature_net)     
        predict_feat(team1, team2, feature_net)     
    input('end')
        


