from scipy.fftpack import fft
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import os, pickle, sys, time

sample_rate = 44100
window_size = 4410
bin_size = sample_rate/window_size
filter_factor = 1.3
fan_out = 5
threshold = 25

bands = [10, 20, 40, 80, 160, 512]
database_addresses = {}
# --- OLD STUFF --- #

def window_samples(samples, start=0, plot=False):
    window = np.hamming(window_size)
    samples_short = samples[start:start+window_size]
    samples_windowed = window * samples_short
    if plot:
        plt.plot(samples_short)
        plt.title('First %s samples' % window_size)
        plt.figure()
        plt.plot(samples_windowed)
        plt.title('First %s samples with window function applied' % window_size)
        plt.show()
    return samples_windowed

def generate_addresses(audio, movie_id=None):
    i = 0
    addresses = {}
    local_peaks_positive = []
    local_peaks_negative = []
    peak1 = (0, -1)
    peak2 = (0, -1)
    ascending = True
    for i, amp in enumerate(audio):

        if i % window_size == 0:
            percent = round(i/len(audio)*100, 1)
            print("Progress %.1f%%" % percent, end="\r")
            if peak1[1] > peak2[1]:
                # swap as peak1 is after peak2
                temp = peak2
                peak2 = peak1
                peak1 = temp

            peak1 = (peak1[1]%window_size)+1
            # peak2 = (peak2[1]%window_size)+1
            peak2 = window_size-(peak2[1]%window_size)+1
            local_peaks_positive.sort(key=lambda x: x[1])
            local_peaks_negative.sort(key=lambda x: x[1])
            top_6_positive = local_peaks_positive[:6]
            top_6_negative = local_peaks_negative[:6]
            top_6_positive.sort()
            top_6_negative.sort()
            for j in range(6):
                try:
                    address = "%d:%d:%d" % (top_6_positive[j][0]*top_6_positive[j*2][0],
                                            top_6_negative[j*2][0]*top_6_negative[j*2+1][0],
                                            top_6_negative[j][0]-top_6_positive[j][0])
                    if movie_id:
                        addresses[address] = (int(i/window_size)*0.1, movie_id)
                    else:
                        addresses[address] = int(i/window_size)*0.1
                except IndexError:
                    continue
            local_peaks_positive = []
            local_peaks_negative = []
            peak1 = (0, -1)
            peak2 = (0, -1)
            ascending = True
            continue
        if not ascending and amp > audio[i-1]:
            ascending = True
            if amp < 0:
                local_peaks_negative.append((i%window_size, audio[i-1]))
        if amp < audio[i-1] and ascending:
            ascending = False
            if amp > 0:
                if amp > peak1[0]:
                    peak2 = peak1
                    peak1 = (amp, i)
                elif amp > peak2[0]:
                    peak2 = (amp, i)
                local_peaks_positive.append((i%window_size, audio[i-1]))
    return addresses

def match(sample_addresses, database_addresses, plot=False):
    bin_dict = {}
    bin_points = {}
    for address_s, time_s in sample_addresses.items():
        tuples_d = database_addresses.get(address_s)
        if tuples_d is None:
            continue
        for tuple_d in tuples_d:
            time_d, movie_id = tuple_d
            time_offset = time_d-time_s
            pair = '%.1f:%d' % (time_offset, movie_id)
            if not bin_dict.get(pair):
                bin_dict[pair] = 0
            try:
                bin_points[movie_id].append((time_s, time_d))
            except:
                bin_points[movie_id] = [(time_s, time_d)]
        bin_dict[pair] += 1

    if plot:
        for movie_id, points in bin_points.items():
            x = [round(k[1], 2) for k in points]
            y = [round(k[0], 2) for k in points]
            plt.clf()
            plt.title('MOVIE ID: %d' % movie_id)
            plt.scatter(x, y, marker='o')
            plt.xlabel('Database soundfile time')
            plt.ylabel('Sample soundfile time')
            plt.show()

    maximum = 0
    max_pair = ''
    for pair, val in bin_dict.items():
        if val > maximum:
            maximum = val
            max_pair = pair
    return maximum, max_pair

def add_to_database(addresses):
    for key, value in addresses.items():
        try:
            database_addresses[key].append(value)
        except KeyError:
            database_addresses[key] = [value]

# --- TESTING --- #

counter = 1
for i in range(2, 3):
    with open('non_filtered_db'+str(i)+'.pickle', 'rb') as f:
        partial_db = pickle.load(f)
    with open('db_key'+str(i)+'.pickle', 'rb') as f:
        db_key = pickle.load(f)
    for j, sample in enumerate(partial_db):
        if not j == 3: continue
        # print('Peaks')
        # db_peaks = generate_peaks(sample)
        # with open('lego_peaks1_new.pickle', 'rb') as f:
        #     db_peaks = pickle.load(f)
        # print('Filtering')
        # filtered_sample = filtering_process(sample, True)
        # print('Filtered:', db_key[j])
        print('Addresses:')
        sample_addresses = generate_addresses(sample, movie_id=counter)
        print('Generated addresses:', db_key[j])
        # if sample_addresses: print('here')
        add_to_database(sample_addresses)
        counter += 1

# lego1, clip_peaks = load_file('clips/09_lego1.wav')
lego1, _ = lr.load('pitch_shifted_clips/lego2_shifted2.wav', sr=sample_rate)
# lego1, _ = lr.load('clips/09_lego2.wav', sr=sample_rate)
# for i in range(-50, 125, 25):
#     new_audio = lr.effects.pitch_shift(lego1, sample_rate, i/100)
#     clip_peaks = generate_peaks(new_audio)
#     filtered_new_audio = filtering_process(new_audio)
#     clip_addresses = generate_addresses(filtered_new_audio, peaks=clip_peaks)
#     # print(clip_addresses)
#     print("Pitch shift:", i/100, "=", match(clip_addresses, database_addresses))
# lego1, clip_peaks = load_file('clips/09_lego3.wav')
# # filtered_clip = filtering_process(lego1)
clip_addresses = generate_addresses(lego1)
print(match(clip_addresses, database_addresses))
