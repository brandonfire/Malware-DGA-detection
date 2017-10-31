#!/usr/bin/env python3
import glob
import statistics
import numpy as np

g_s_search_string = "Cryptolocker"
g_s_path = "./*.txt"  # Location of log files
g_i_cache_threshold = 100  # Number of Domain to Conduct Averages
g_i_detection_threshold = 1.0  # Sensitivty of Match

g_sz_files = glob.glob(g_s_path)
g_i_dga_counter = 0
g_sz_dga_cache = []
g_i_smallest_dga_length = 0
g_sz_dga_detection = []
g_sz_dga_detection_stdev = []
g_r_dga_result = []

for s_file in g_sz_files:

    f = open(s_file, "r")

    for s_line in iter(f):
        s_line = s_line.upper()
        if g_s_search_string.upper() in s_line:
            s_line = s_line.upper()
            if g_i_dga_counter < g_i_cache_threshold:
                sz_dga_in_loop = []

            # Processes entry in data set and removes non-useful information
            #print(s_line.rstrip('\n').split(',')[0].split('.')[0])
            for c in s_line.rstrip('\n').split(',')[0].split('.')[0]:

                if g_i_dga_counter < g_i_cache_threshold:
                    sz_dga_in_loop.append(ord(c))
            print(sz_dga_in_loop)
            # Sets the initial length size to observed value
            if g_i_dga_counter < 1:
                g_i_smallest_dga_length = len(sz_dga_in_loop)

            # Calculates the Detection Signature
            if g_i_dga_counter == g_i_cache_threshold:
                for i in range(g_i_smallest_dga_length):
                    d_ord_sum_total = 0
                    sz_dga_detection_stdev = []
                    for j in g_sz_dga_cache:
                        d_ord_sum_total += j[i]
                        sz_dga_detection_stdev.append(j[i])
                    g_sz_dga_detection_stdev.append(
                        statistics.stdev(sz_dga_detection_stdev)
                    )
                    g_sz_dga_detection.append(
                        round(d_ord_sum_total / g_i_cache_threshold)
                    )

            # Stores DGA in Cache for Calculating Detection
            # if Less Than Cache Threshold
            if g_i_dga_counter < g_i_cache_threshold:
                g_sz_dga_cache.append(sz_dga_in_loop)

                if g_i_smallest_dga_length > len(sz_dga_in_loop):
                    g_i_smallest_dga_length = len(sz_dga_in_loop)
            else:
                i_detection_score = 0
                for i in range(g_i_smallest_dga_length):
                    if (g_sz_dga_detection[i] - g_sz_dga_detection_stdev[i]) \
                            <= sz_dga_in_loop[i] <= \
                            (g_sz_dga_detection[i] + g_sz_dga_detection_stdev[i]):
                        i_detection_score += 1
                g_r_dga_result.append(i_detection_score)

            g_i_dga_counter += 1

    f.close()

# Totaling Results
g_i_total_result = 0
for i in g_r_dga_result:
    if (i/len(g_sz_dga_detection)) >= g_i_detection_threshold:
        g_i_total_result += 1

print("DGA SIGNATURE: " + str(g_sz_dga_detection))
print("STDEV: " + str(g_sz_dga_detection_stdev))
print("Detection Threshold: " + str(g_i_detection_threshold))
print("Cache Size: " + str(g_i_cache_threshold))
print("Positive Threat: " + str(g_i_total_result))
print("Total: " + str(g_i_dga_counter))
print("Result: " + str(g_i_total_result/g_i_dga_counter))
