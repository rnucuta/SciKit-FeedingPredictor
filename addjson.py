import json
f = open("data.json", "r")
final = json.load(f)
f.close()
f = open("data.json", "w+")

final["timesFed"] = [[945,1215,1222,1228,1232,1237,1239,1307,1338,1339,1351,1359,1401,1419,1420,1433,1452,1522,1526,1532,1541,1600],[1012,1021,1022,1040,1055,1117,1143,1225,1231,1327,1328,1334,1339,1342,1419], [923,940,957,959,1002,1018,1024,1029,1055,1119,1129,1139,1146,1152,1220,1225,1238,1307,1322,1404,1553,1558]]
# for i in range(len(final["before"])-1, 2, -1):
# 	final['before'][i]=final['before'][i-3]
# final["before"][0]=0
# final["before"][1]=0
# final["before"][2]=0
print(final)
json.dump(final, f)
f.close()