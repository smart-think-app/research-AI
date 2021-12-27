import pandas as pd
from scipy import spatial
import operator
import numpy as np

df_location = pd.read_excel("../DataTrain.xlsx", sheet_name="location_genres")
m_population = {
    "high": 2,
    "medium": 1,
    "low": 0
}


def compute_distance2(a, b):
    distance = spatial.distance.cosine(a, b)
    return distance


def compute_distance(a, b):  # more lower more similar
    genres_sod_A = a[0]
    genres_sod_B = b[0]
    distance_sod = spatial.distance.minkowski(genres_sod_A, genres_sod_B)

    genres_age_A = a[2]
    genres_age_B = b[2]
    distance_age = spatial.distance.cosine(genres_age_A, genres_age_B)

    return distance_sod + distance_age + abs(a[1] - b[1])


df_location["population"] = df_location["population"].map(m_population)
location_dict = {}
location_arr = []
for index, r in df_location.iterrows():
    location_arr.append(r["location"])
    location_dict[r["location"]] = (
        (r["mean_sod"], r["median_sod"], r["mode_sod"]), r["population"],
        (r["mean_age"], r["median_age"], r["mode_age"])
    )


def get_neighbors(buyerId, k):
    distances_arr = []
    for buyer in location_dict:
        if buyer != buyerId:
            dist = compute_distance(location_dict[buyerId], location_dict[buyer])
            distances_arr.append((buyer, dist))
    distances_arr.sort(key=operator.itemgetter(1))

    neighbors_arr = []
    for x in range(k):
        neighbors_arr.append(distances_arr[x][0])

    return neighbors_arr


corr_matrix = []
for i in location_arr:
    list_neighbors = get_neighbors(i, 3)
    corr_matrix.append([i, ",".join(list_neighbors)])
df_new = pd.DataFrame(np.array(corr_matrix), columns=["location", "similar_location"])
print(df_new)
df_new.to_excel("../similar_location.xlsx")
# K = 3
# neighbors = get_neighbors("HN", K)
# print(neighbors)
