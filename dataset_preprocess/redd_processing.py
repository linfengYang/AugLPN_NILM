import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

params_appliance = {
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2, 3],
        'channels': [11, 6, 16],
        'train_build': [2, 3],
        'test_build': 1
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2, 3],
        'channels': [5, 9, 7],
        'train_build': [2, 3],
        'test_build': 1
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2, 3],
        'channels': [6, 10, 9],
        'train_build': [2, 3],
        'test_build': 1
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3],
        # 'channels': [19, 7, 13],
        'channels': [20, 7, 13],
        'train_build': [2, 3],
        'test_build': 1
    }
}
aggregate_mean = 522
aggregate_std = 814
start_time = time.time()
data_dir = 'low_freq'  # REDD dataset path
sample_seconds = 6
validation_percent = 10

nrows = None

appliance_names = ["microwave", "fridge", "dishwasher", "washingmachine"]

for appliance_name in appliance_names:
    print('--' * 20)
    print('\n' + appliance_name)
    train = pd.DataFrame(columns=['aggregate', appliance_name])
    save_path = 'created_data/REDD/{}/'.format(appliance_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for h in params_appliance[appliance_name]['houses']:   # [1,2,3]
        print('    ' + data_dir + '/house_' + str(h) + '/'
              + 'channel_' +
              str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
              '.dat')

        mains1_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                  str(1) + '.dat',
                                  sep="\s+",
                                  nrows=nrows,
                                  usecols=[0, 1],
                                  names=['time', 'mains1'],
                                  dtype={'time': str},
                                  )

        mains2_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                  str(2) + '.dat',
                                  sep="\s+",
                                  nrows=nrows,
                                  usecols=[0, 1],
                                  names=['time', 'mains2'],
                                  dtype={'time': str},
                                  )
        app_df = pd.read_table(data_dir + '/' + 'house_' + str(h) + '/' + 'channel_' +
                               str(params_appliance[appliance_name]['channels']
                                   [params_appliance[appliance_name]['houses'].index(h)]) + '.dat',
                               sep="\s+",
                               nrows=nrows,
                               usecols=[0, 1],
                               names=['time', appliance_name],
                               dtype={'time': str},
                               )

        mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
        mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')

        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)

        mains_df = mains1_df.join(mains2_df, how='outer')

        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
        # resample = mains_df.resample(str(sample_seconds) + 'S').mean()

        mains_df.reset_index(inplace=True)

        # deleting original separate mains
        del mains_df['mains1'], mains_df['mains2']

        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        # 索引都是时间的，做外连接，时间进行合并
        df_align = mains_df.join(app_df, how='outer'). \
            resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)

        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)
        # print(df_align.count())
        # df_align['OVER 5 MINS'] = (df_align['time'].diff()).dt.seconds > 9
        # df_align.plot()
        # plt.plot(df_align['OVER 5 MINS'])
        # plt.show()

        del mains1_df, mains2_df, mains_df, app_df, df_align['time']

        mains = df_align['aggregate'].values
        app_data = df_align[appliance_name].values
        # plt.plot(np.arange(0, len(mains)), mains, app_data)
        # plt.show()

        # if debug:
        #         # plot the dtaset
        #     print("df_align:")
        #     print(df_align.head())
        #     plt.plot(df_align['aggregate'].values)
        #     plt.plot(df_align[appliance_name].values)
        #     plt.show()

        # Normolization
        mean = params_appliance[appliance_name]['mean']
        std = params_appliance[appliance_name]['std']


        df_align['aggregate'] = (df_align['aggregate'] - aggregate_mean) / aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std


        if h == params_appliance[appliance_name]['test_build']:
            # Test CSV
            df_align.to_csv(save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)
            print("    Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
            continue

        train = train.append(df_align, ignore_index=True)
        del df_align

    # Validation CSV
    val_len = int((len(train) / 100) * validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    val.to_csv(save_path + appliance_name + '_validation_' + '.csv', mode='a', index=False, header=False)

    # Training CSV
    train.drop(train.index[-val_len:], inplace=True)
    train.to_csv(save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)

