import sys
# txtStrings = modelDavis.split("\n")

# model D 0.6:
# total: 837
# right: 730
# wrong: 65
# unknown: 42


# acc: 87.21624850657109

# davis 0.4
# total: 837
# right: 729
# wrong: 56
# unknown: 52

def getNiumAccuracy(sampleList):
    right = 0
    wrong = 0
    uk = 0
    total = 0
    acc = 0
    for stri in sampleList:
        if len(stri.split(",")) != 3:
            continue
        total += 1
        # print(stri)
        print(stri.split(","))
        filename,dist,res = map(str.strip, stri.split(","))
        filename = filename.split("_")[5].split("/")[1]
        print(filename, dist, res)
        # print(f"filename: {filename} dist: {dist} res: {res}")
        if filename == res:
            right += 1
        elif res == "Unknown":
            uk += 1
        elif filename != res and res != "Unknown":
            wrong += 1
    
    print(f"total: {total}\nright: {right}\nwrong: {wrong}\nunknown: {uk}")
    acc = (right / total) * 100
    print(f"\n\nacc: {acc}")
    return (total, right, wrong, uk, acc)

def getLFWAccuracy(sampleList):
    right = 0
    wrong = 0
    uk = 0
    total = 0
    
    for currStr in sampleList:
        currData = currStr.split("/")[-1]
        currData = currData.split(",")
        if len(currData) != 3:
            continue
        actualLabel = currData[0].split(".")[0][:-5].strip()
        dist = currData[1]
        regName = currData[2].strip()
        # print(currData)
        # print(currData[0])
        # print(f"regName: {regName} dist: {dist} actualLabel: {actualLabel}")

        print(actualLabel, dist, regName)
        
        total+=1
        if actualLabel == regName:
            right +=1
            # print("right!")
        elif regName == "Unknown".strip():
            uk +=1
        elif regName != actualLabel:
            print(f"{regName} != {actualLabel} wrong!")
            wrong +=1
        
    print(f"\ntotal: {total}\nright: {right}\nwrong: {wrong}\nunknown: {uk}\n\nsum: {right+wrong+uk}")
    acc = (right / total) * 100
    print(f"accuracy: {acc}")
    return (total, right, wrong, uk, acc)

# acc: 87.09677419354838
if len(sys.argv) != 3:
    print("Run the script with the dataSet name, and file name as arguments.")
_, dataSet, fileName = sys.argv


sampleList = []
with open(fileName) as f:
    sampleList = f.readlines()

if dataSet == "nium":
    print(getNiumAccuracy(sampleList))
elif dataSet == "lfw":
    print(getLFWAccuracy(sampleList))
else:
    print("unknown dataset.")

# /home/ishaan/saralweb/work/projects/NIUM_data/ver_test_data_flat_fname/3065_bd0492ad-ccdb-4781-9eee-35fc0edd07a6.png,0.435839,Unknown



# sampleStr = '''/home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Colin_Powell_0027.jpg,0.358708,Colin_Powell
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Ariel_Sharon_0032.jpg,0.345897,Ariel_Sharon
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/George_W_Bush_0401.jpg,0.305387,Saddam_Hussein
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Mark_Philippoussis_0010.jpg,0.570307,Mark_Philippoussis
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Jan_Ullrich_0005.jpg,0.465585,Jan_Ullrich
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Jacques_Chirac_0022.jpg,0.412807,Jacques_Chirac
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Hugo_Chavez_0020.jpg,0.405841,Hugo_Chavez
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Arnold_Schwarzenegger_0021.jpg,0.366002,Arnold_Schwarzenegger
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Bill_Simon_0009.jpg,0.477459,Bill_Simon
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Roh_Moo-hyun_0016.jpg,0.265745,Roh_Moo-hyun
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Jose_Serra_0001.jpg,0.422319,Jose_Serra
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Tariq_Aziz_0006.jpg,0.510406,Tariq_Aziz
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Jason_Kidd_0004.jpg,0.573277,Jason_Kidd
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Laura_Bush_0027.jpg,0.543378,Laura_Bush
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Jimmy_Carter_0004.jpg,0.432633,Jimmy_Carter
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/John_Kerry_0007.jpg,0.400734,John_Kerry
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Colin_Powell_0192.jpg,0.351847,Colin_Powell
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Winona_Ryder_0010.jpg,0.498345,Winona_Ryder
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Paul_Bremer_0016.jpg,0.560691,Paul_Bremer
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Tom_Daschle_0002.jpg,0.300011,Tom_Daschle
# /home/ishaan/saralweb/work/projects/LFW_data/lfw_data_test_recog/Calista_Flockhart_0001.jpg,0.546701,Calista_Flockhart'''

# sampleList = sampleStr.split("\n")



# ssList[-1].split(",")

# print(14/15 * 100)