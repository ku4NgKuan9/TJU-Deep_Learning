import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

csv_data = pd.read_csv('coupled-data-Z.csv')
datas = csv_data.values  # 转换为 numpy 数组
np.random.shuffle(datas) # 打乱数组顺序
# ！！输入训练集数量 ！！
train_num = 184
Xtrain = datas[:train_num, 3:6]  #训练集X
Ytrain = datas[:train_num, 0:3]  #训练集Y
Xtest = datas[train_num:, 3:6]  #测试集x
Ytest = datas[train_num:, 0:3]  #测试集Y

# 对训练数据进行归一化处理
scaler_X = MinMaxScaler(feature_range=(-1, 1))
Xtrain_scaled = scaler_X.fit_transform(Xtrain)
Xtest_scaled = scaler_X.transform(Xtest)

scaler_Y = MinMaxScaler(feature_range=(-1, 1))
Ytrain_scaled = scaler_Y.fit_transform(Ytrain)
Ytest_scaled = scaler_Y.transform(Ytest)


# 假设 input_features 是输入层的特征数量
input_features = 3

# 创建顺序模型
model = Sequential([
    # 输入层和第一个隐藏层，假设输入数据的特征数为3
    Dense(40, activation='relu', input_shape=(input_features,)),
    Dense(60, activation='relu'),
    Dense(80, activation='relu'), Dense(120, activation='relu'), Dense(160, activation='relu'),
    Dense(160, activation='relu'), Dense(120, activation='relu'), Dense(80, activation='relu'),
    Dense(60, activation='relu'),
    Dense(40, activation='relu'),
    # 输出层，假设输出数据有3个特征
    Dense(3, activation='linear')
])

# 编译模型，使用'adam'优化器和'mean_squared_error'作为损失函数
model.compile(optimizer='adam', loss='mean_squared_error')
# 监控验证集上的损失（val_loss），如果连续10个epoch没有改善则停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

# 训练模型，epochs 表示训练迭代的次数，batch_size 表示每批处理的样本数量
history = model.fit(
    Xtrain_scaled, Ytrain_scaled,
    epochs=500,
    batch_size=32,
    validation_split=0.2,  # 划分20%的训练数据作为验证集
    callbacks=[early_stopping],  # 将早停法回调传递给模型训练
    verbose=1
)
if early_stopping.stopped_epoch:
    print("Training stopped at epoch:", early_stopping.stopped_epoch)

Ytrainpred = model.predict(Xtrain_scaled)   #训练集预测值
Ytestpred = model.predict(Xtest_scaled)     #测试集预测值
# 将预测结果反归一化回原始尺度
Ytrainpred_original_scale = scaler_Y.inverse_transform(Ytrainpred)
Ytestpred_original_scale = scaler_Y.inverse_transform(Ytestpred)

print(Ytestpred_original_scale)
print(Ytest)

# 导出数据
df_Ytestpred_original_scale = pd.DataFrame(Ytestpred_original_scale)
df_Ytest = pd.DataFrame(Ytest)
df_Ytestpred_original_scale.columns = ['Column1', 'Column2', 'Column3']  # 根据您的数据调整列名和数量
df_Ytest.columns = ['Column1', 'Column2', 'Column3']
df_Ytestpred_original_scale.to_csv('Ytestpred_original_scale.csv', index=False)
df_Ytest.to_csv('Ytest.csv', index=False)