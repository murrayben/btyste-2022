from scipy.fftpack import fft
import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import os, pickle, sys, time

sample_rate = 44100
window_size = 4096
bin_size = sample_rate/window_size
filter_factor = 1.3
fan_out = 5
threshold = 25

bands = [10, 20, 40, 80, 160, 512]
database_addresses = {}


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

def fft_analysis(samples, plot=False):
    length = int(len(samples))
    fft_result = fft(samples, window_size)
    if plot:
        values = np.arange(length)
        frequencies = values/(length/sample_rate)
        plt.title('FFT')
        plt.plot(frequencies[:(window_size//2)], abs(fft_result)[:(window_size//2)])
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.show()
    return fft_result

def filtering_process(audio, plot=False):
    powerful_bins = []
    count = 0
    for i in range(0, len(audio)+1, window_size):
        if len(audio) + 1 - i < window_size:
            break
#             i = len(audio)-window_size
        try:
            window = window_samples(audio, start=i)
        except:
            print(i, len(audio))
            continue
        bins = fft_analysis(window)
        current_bin = 0
        local_max = 0
        local_max_bin = None
        j = 0
        curr_time = i/window_size * (window_size/sample_rate)
        while current_bin < bands[-1]:
            if abs(bins[current_bin]) > local_max:
                local_max = abs(bins[current_bin])
                local_max_bin = current_bin
            current_bin += 1
            if current_bin == bands[j]:
                j += 1
                if local_max_bin is not None:
                    powerful_bins.append(
                        (local_max,(local_max_bin+1)*(sample_rate/window_size),curr_time)
                    )
                local_max = 0
                local_max_bin = None
        count += window_size/len(audio)*100
        percent = str(round(count)) + "%"
        print("Progress {0}".format(percent), end="\r")
    average = sum([i[0] for i in powerful_bins])/len(powerful_bins)
    filtered_bins = []
    average *= filter_factor
    for k in powerful_bins:
        if k[0] >= average:
            filtered_bins.append(k)
    
    if plot:
        x = [ele[2] for ele in filtered_bins]
        y = [ele[1] for ele in filtered_bins]
        plt.figure()
        plt.scatter(x, y, marker='x')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.plot()
    return filtered_bins

def load_file(filename, plot=False, measure_time=True):
    if measure_time:
        time_start = time.time()
    sound, _ = lr.load('./{0}'.format(filename), sr=sample_rate)

    filtered_sound = filtering_process(sound, plot=plot)
    if measure_time:
        time_end = time.time()
        elasped = time_end-time_start
        print('%s took %.2f secs' % (filename, elasped))
    return filtered_sound

def generate_addresses(filtered_sound, movie_id=None):
    sorted_sound = [(round(k[2], 1), round(k[1], 2)) for k in filtered_sound]
    sorted_sound.sort()

    i = 0
    addresses = {}
    finished = False
    while i < len(sorted_sound) and not finished:
        j = 1
        if len(sorted_sound)-(i+1) < fan_out:
            i = len(sorted_sound)-fan_out-1
            finished = True
        while j <= fan_out:
            if movie_id is not None:
                address_tuple = (sorted_sound[i][0], movie_id)
            else:
                address_tuple = sorted_sound[i][0]
            addresses["%.2f:%.2f:%.1f" %
                (
                    sorted_sound[i][1],
                    sorted_sound[i+j][1],
                    sorted_sound[i+j][0]-sorted_sound[i][0]
                )
            ] = address_tuple
            j+=1
        i+=2

    return addresses

def match(sample_addresses, database_addresses):
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

samples = []
if len(sys.argv) > 1 and sys.argv[1] == '--samples':
    start_time = time.time()
    os.chdir(r'C:\projects\shazam_btyste\shazam_algorithm\clips')
    counter = 1
    for f in os.listdir():
        f_name, f_ext = os.path.splitext(f)
        if counter > 10: break
        if f_ext == '.wav':
            sample = load_file(f_name.strip() + f_ext.strip())
            movie_id = int(f_name[:2])
            samples.append((sample, counter, movie_id))
            counter += 1
    os.chdir(r'C:\projects\shazam_btyste\shazam_algorithm')
    with open('samples_final_movies_new.pickle', 'wb') as f:
        pickle.dump(samples, f)
    end_time = time.time()
    taken = end_time-start_time
    mins = taken // 60
    secs = taken % 60
    print('Files loaded successfully in %d mins and %d secs' % (mins, secs))
    sys.exit(0)

sample_add = {}
with open('samples_final_movies_new.pickle', 'rb') as f:
    samples = pickle.load(f)

counter = 1
for i in range(1, 11):
    with open('non_filtered_db'+str(i)+'.pickle', 'rb') as f:
        partial_db = pickle.load(f)
    for j, sample in enumerate(partial_db):
        print('Filtering MOVIE ID', counter)
        filtered_sample = filtering_process(sample)
        sample_addresses = generate_addresses(filtered_sample, movie_id=counter)
        add_to_database(sample_addresses)
        counter += 1

for sample, counter, movie_id in samples:
    print('Sample:', counter)
    sample_add["%d %d" % (counter, movie_id)] = generate_addresses(sample)
print('Addresses generated successfully')


correct = 0
total = 0
for pair, addresses_s in sample_add.items():
    id_s, actual_movie = map(int, pair.split())
    print("Sample clip {0}... - movie ID: {1}".format(id_s, actual_movie))
    count, max_pair = match(addresses_s, database_addresses)
    offset, movie_id = max_pair.split(':')
    offset = float(offset)
    movie_id = int(movie_id)
    if count >= threshold:
        if actual_movie == movie_id:
            correct += 1
            print('--- Correct - was in database')
        else:
            print('--- Incorrect - false positive')
    else:
        if actual_movie == 0:
            correct += 1
            print('--- Correct - was not in database')
        else:
            print('--- Incorrect - false negative')
    print('--- Count:', count, 'Offset:', offset, 'Movie ID', movie_id)
    total += 1

print('Accuracy:', round((correct/total)*100, 2), 'out of a total of', total, 'clips.')


