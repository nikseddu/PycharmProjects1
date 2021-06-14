import json
import os


Data_dir = r'Data'

final_data=[]
#
for ifile_name in os.listdir(Data_dir):

    if ifile_name.endswith(".json"):
        print(ifile_name)
        file_path=Data_dir+"/"+ifile_name

        with open(file_path,"r") as file:
            json_data = json.load(file)
            print(file_path)
            final_data.extend(json_data)

print(final_data[0])


annotation_result = final_data[0]['annotations'][0]['result']
total_list=[]

for i in annotation_result:
    entity_list = []
    # print(i['value'], "\n")
    values = i['value']
    # print(values["text"])
    #
    # print(i['text'], "\n")
    entity_list.append([values['start'],values['end'],values['labels'][0]])
    total_list.append((values['text'], {'entities': entity_list}))
#     # print("\n")

print(total_list)








# data1 = data[0]
#
# # print(data1["annotations"])
#
# training_data = []
#

# annotation_result = data1['annotations'][0]['result']

# print(annotation_result)






# for r in annotation_result:
#
#         r.pop('id')
#         # r.pop('from_name')
#         # r.pop('to_name')
#         # r.pop('type')

# print(annotation_result)

# annotation_result["value"]




#
# Spacy Format
# train =[("Money Transfer from my checking account is not working",{"entities":[(6,13,"ACTIVITY"),(23,39,"PRODUCT")]}),
#         ("I want to check balance in saving account",{"entities":[(16,23,"ACTIVITY"),(30,45,"PRODUCT")]})
# ]



# with open("Data/194.json", "r") as f:
#     lines = f.readlines()
#
# for line in lines:
#     data  =json.loads(line)
#
#     for result in data['result']:
#         # only a single point in text annotation.
#         point = result['value'][0]
#         print("hey")


# final_data = json.load()

# total_list=[]

# for j,i in enumerate(final_data):
# #     entity_list=[]
# #     try:
# #         annotation_result = final_data['annotations'][0]['result']
# #         for x in annotation_result['annotations']:
# #             entity_list.append([x['start'],x['end'],x['labels'][0]])
# #         total_list.append((i['text'],{'entities':entity_list}))
# #     except:
# #         print(j)
    # print(i)
    # # print(j)
    # print(i["annotations"])
# print(total_list)

# annotation_result = final_data[0]['annotations'][0]['result']


# for i in annotation_result:
#     entity_list = []
#     # print(i['value'], "\n")
#     values = i['value']
#     # print(values["text"])
#     #
#     # print(i['text'], "\n")
#     entity_list.append([values['start'],values['end'],values['labels'][0]])
#     total_list.append((values['text'], {'entities': entity_list}))
# #     # print("\n")
#
# print(total_list)
# print(type(annotation_result))