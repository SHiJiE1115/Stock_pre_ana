# model.py
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Nadam

# ==================== 模型构建模块 ====================
def create_cnn_lstm(input_shape):
    inputs = Input(shape=input_shape)

    # CNN部分
    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # LSTM部分
    x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)
    x = LSTM(64, dropout=0.2, recurrent_dropout=0.1)(x)

    # 全连接层
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Nadam(learning_rate=0.0005),
        loss='huber_loss',
        metrics=['mae']
    )
    return model
