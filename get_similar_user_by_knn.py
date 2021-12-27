import pandas as pd
from scipy import spatial
import operator


def compute_distance(a, b):
    genresA = a[3]
    genresB = b[3]
    distance = spatial.distance.cosine(genresA, genresB)
    return distance + a[0] + a[1] + a[2]


df = pd.read_excel("Products.xlsx", sheet_name="BuyerGenres")
mapLocation = {
    "HCM": 1,
}
mapGender = {
    "Male": 1,
}
df["Location"] = df["Location"].map(mapLocation)
df["Gender"] = df["Gender"].map(mapGender)

buyer_dict = {}
for index, r in df.iterrows():
    # print(r[4:8])
    buyer_dict[r["UserId"]] = (r["UserId"], r["Location"], r["Gender"],
                               [r["Technology"], r["Beauty"], r["Travel"], r["Fashion"]])


def get_neighbors(buyerId, k):
    distances_arr = []
    for buyer in buyer_dict:
        if buyer != buyerId:
            dist = compute_distance(buyer_dict[buyerId], buyer_dict[buyer])
            distances_arr.append((buyer, dist))
    distances_arr.sort(key=operator.itemgetter(1))

    neighbors_arr = []
    for x in range(k):
        neighbors_arr.append(distances_arr[x][0])

    return neighbors_arr


K = 3
neighbors = get_neighbors(1, K)
print(neighbors)
