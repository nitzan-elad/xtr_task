import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


class Sample:
    def __init__(self, file_path, number_of_channels, freq=4000):
        self.file_path = file_path
        self.number_of_channels = number_of_channels
        self.freq = freq
        self.data = None

    def convert_to_df(self):
        try:
            self.data = pd.DataFrame(self.data)
            return self.data
        except Exception as e:
            print(f'Error converting to DataFrame: {e}')

    def load_file(self, num_ADC_bits=15, voltage_resolution=4.12e-7):
        try:
            data = np.fromfile(self.file_path, dtype='uint16')
            data = np.reshape(data, (self.number_of_channels, -1), order='F')
            data = np.multiply(voltage_resolution, (data - np.float_power(2, num_ADC_bits - 1)))
            self.data = data
            return self.data
        except Exception as e:
            print(f'Error loading file: {e}')

    def bandpass(self, lowcut, highcut, order=2):
        nyq = self.freq / 2
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band', analog=False,output='ba',fs=self.freq)
        return b, a

    def apply_bandpass(self, lowcut, highcut, order=2):
        b, a = self.bandpass(lowcut, highcut, order=order)
        try:
            self.data = filtfilt(b, a, self.data, axis=1)
            return self.data
        except Exception as e:
            print(f'Error applying bandpass filter: {e}')

    def plot_data(self):
        start = 0
        end = self.data.shape[1] / self.freq #total time of the sampling
        dt = 1/self.freq
        time = np.arange(start, end, dt)
        plt.figure(figsize=[12,9])

        for i in range(self.number_of_channels):
            plt.subplot(self.number_of_channels,1,i+1)
            plt.plot(time,self.data[i,:],label=f'Channel {i+1}')
            plt.title(f'Ch{i+1}', loc='left')
            plt.xlabel('[s]')
            plt.ylabel('[V]', loc='center')

        plt.tight_layout()
        plt.show()


path = 'C:\\Users\\nitza\\Downloads\\NEUR0000.DT8'
sample = Sample(path, 8)

sample.load_file()
sample.convert_to_df()
print(sample.data)
print(type(sample.data))

sample.apply_bandpass(200, 400)
print(type(sample.data))
sample.plot_data()
