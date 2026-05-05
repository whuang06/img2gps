import numpy as np

ref_coords = np.load("ref_coords.npy")

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 39.950375, 39.95277777777778, -75.19258888888889, -75.17618611111112
BINS = 3

def get_label(lat, lon):
    row = int(((lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * BINS)
    col = int(((lon - MIN_LON) / (MAX_LON - MIN_LON)) * BINS)
    row = max(0, min(row, BINS - 1))
    col = max(0, min(col, BINS - 1))
    return row + (col * BINS)

ref_bins = np.array([get_label(c[0], c[1]) for c in ref_coords])
np.save("ref_bins.npy", ref_bins)
print("Reference bins saved!")