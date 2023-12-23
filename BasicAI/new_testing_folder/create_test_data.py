import random
import csv

#Creating test data
data_list = [["r1", "r2", "r3", "label"]]

while True:
        if len(data_list) == 3500:
            break

        r1 = random.randint(-100, 100)
        r2 = random.randint(-100, 100)
        r3 = random.randint(-100, 100)
        s = sum([r1, r2, r3]) #this is also the label
        data_row = [r1, r2, r3, s]

        if data_row in data_list:
            continue
        else:
            data_list.append(data_row)



with open("new_testing_folder/test_data.csv", "w") as fila:
    writer = csv.writer(fila)

    for row in data_list:
        writer.writerow(row)
        
    




        



