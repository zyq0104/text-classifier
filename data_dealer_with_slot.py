import numpy as np

data_intent = "./intent_datas"
data_slot = "./slot_datas"
data_seg = "./seg_datas"
data_out_label1 = "./train_label1_data_with_slot"
data_out_label2 = "./train_label2_data_with_slot"
data_out_label3 = "./train_label3_data_with_slot"

intent = open(data_intent, "r").readlines()
slot = open(data_slot, "r").readlines()
seg = open(data_seg, "r").readlines()

out_label1 = open(data_out_label1, "w")
out_label2 = open(data_out_label2, "w")
out_label3 = open(data_out_label3, "w")
dic = {}
pos = 0

for i in slot:
    i = i.strip()
    split = i.split("@")
    for j in split:
        if j not in dic and j != " ":
            # print(j)
            dic[j] = pos
            pos = pos + 1


for i in range(len(intent)):
    # print(intent[i])
    # print(slot[i])
    # print(seg[i])
    slot[i] = slot[i].strip()
    intent[i] = intent[i].strip()
    intent_split = intent[i].split("@")
    slot_split = slot[i].split("@")
    # seg_split = seg[i].split("@")

    slot_value = np.zeros(shape=[37], dtype=np.int32)
    if intent_split[1] == '搜索':
        # print(intent_split[0])
        for j in slot_split:
            slot_value[dic[j]] = 1
            # print(slot_value)
        out_label1.write(intent_split[0] + "@")
        out_label1.write(str(slot_value))
        out_label1.write("\n")
    elif intent_split[1] == '复购':
        # print(intent_split[0])
        for j in slot_split:
            slot_value[dic[j]] = 1
            # print(slot_value)
        out_label2.write(intent_split[0] + "@")
        out_label2.write(str(slot_value))
        out_label2.write("\n")
    elif intent_split[1] == '其他':
        # print(intent_split[0])
        for j in slot_split:
            slot_value[dic[j]] = 1
            # print(slot_value)
        out_label3.write(intent_split[0] + "@")
        out_label3.write(str(slot_value))
        out_label3.write("\n")


data_label1_train = "./final/data_label1_final_output.txt"
data_label2_train = "./final/data_label2_final_output.txt"
data_label3_train = "./final/data_label3_final_output.txt"

test_label1 = "./final/test_data_label1_final_output.txt"
test_label2 = "./final/test_data_label2_final_output.txt"
test_label3 = "./final/test_data_label3_final_output.txt"

label1_train = open(data_label1_train, "r").readlines()
label2_train = open(data_label2_train, "r").readlines()
label3_train = open(data_label3_train, "r").readlines()
label1_test = open(test_label1, "r").readlines()
label2_test = open(test_label2, "r").readlines()
label3_test = open(test_label3, "r").readlines()

dic = {
            "B-poi_name": 1, "I-poi_name": 2,
            "B-poi_category": 3, "I-poi_category": 4,
            "B-product_name": 5, "I-product_name": 6,
            "B-food_classify": 7, "I-food_classify": 8,
            "B-ingredient": 9, "I-ingredient": 10,
            "B-method": 11, "I-method": 12,
            "B-food_amount": 13, "I-food_amount": 14,
            "B-product_sales": 15, "I-product_sales": 16,
            "B-neg_intent": 17, "I-neg_intent": 18,
            "B-poi_comment": 19, "I-poi_comment": 20,
            "B-discount": 21, "I-discount": 22,
            "B-taste": 23, "I-taste": 24,
            "B-dish_price": 25, "I-dish_price": 26,
            "B-deliv_time": 27, "I-deliv_time": 28,
            "B-label2_time": 29, "I-label2_time": 30,
            "B-ref_pro": 31, "I-ref_pro": 32,
            "B-ref_res": 33, "I-ref_res":34,
            "B-deliv_price": 35, "I-deliv_price": 36,
            "B-error_area": 37, "I-error_area": 38

}
dic_reverse = dict(zip(dic.values(), dic.keys()))

output_label1_train = open("./final_2/data_label1_train", "w")
output_label2_train = open("./final_2/data_label2_train", "w")
output_label3_train = open("./final_2/data_label3_train", "w")
output_label1_test = open("./final_2/data_label1_test", "w")
output_label2_test = open("./final_2/data_label2_test", "w")
output_label3_test = open("./final_2/data_label3_test", "w")

slot_value_default = 1
slot_value_default_none = 0

for i in label1_train:
    slot_value = np.array([slot_value_default_none]*18)
    i = i.split("/")
    # print(i[0])
    # print(dic_reverse[1])
    for j in range(16):
        k = j*2+1
        if dic_reverse[k] in i[2]:
            slot_value[j] = slot_value_default
            # print(dic_reverse[k])
    output_label1_train.write(i[0] + "@")
    output_label1_train.write(str(slot_value))
    output_label1_train.write("\n")

for i in label2_train:
    slot_value = np.array([slot_value_default_none] * 18)
    i = i.split("/")
    # print(i[0])
    # print(dic_reverse[1])
    for j in range(16):
        k = j*2+1
        if dic_reverse[k] in i[2]:
            slot_value[j] = slot_value_default
            # print(dic_reverse[k])
    output_label2_train.write(i[0] + "@")
    output_label2_train.write(str(slot_value))
    output_label2_train.write("\n")


for i in label3_train:
    slot_value = np.array([slot_value_default_none]*18)
    i = i.split("/")
    # print(i[0])
    # print(dic_reverse[1])
    for j in range(16):
        k = j*2+1
        if dic_reverse[k] in i[2]:
            slot_value[j] = slot_value_default
            # print(dic_reverse[k])
    output_label3_train.write(i[0] + "@")
    output_label3_train.write(str(slot_value))
    output_label3_train.write("\n")

for i in label1_test:
    slot_value = np.array([slot_value_default_none]*18)
    i = i.split("/")
    # print(i[0])
    # print(dic_reverse[1])
    for j in range(16):
        k = j * 2 + 1
        if dic_reverse[k] in i[2]:
            slot_value[j] = slot_value_default
            # print(dic_reverse[k])
    output_label1_test.write(i[0] + "@")
    output_label1_test.write(str(slot_value))
    output_label1_test.write("\n")

for i in label2_test:
    slot_value = np.array([slot_value_default_none]*18)
    i = i.split("/")
    # print(i[0])
    # print(dic_reverse[1])
    for j in range(16):
        k = j * 2 + 1
        if dic_reverse[k] in i[2]:
            slot_value[j] = slot_value_default
            # print(dic_reverse[k])
    output_label2_test.write(i[0] + "@")
    output_label2_test.write(str(slot_value))
    output_label2_test.write("\n")

for i in label3_test:
    slot_value = np.array([slot_value_default_none] * 18)
    i = i.split("/")
    # print(i[0])
    # print(dic_reverse[1])
    for j in range(16):
        k = j * 2 + 1
        if dic_reverse[k] in i[2]:
            slot_value[j] = slot_value_default
            # print(dic_reverse[k])
    output_label3_test.write(i[0] + "@")
    output_label3_test.write(str(slot_value))
    output_label3_test.write("\n")
