import csv
import numpy as np
import sys

filename = sys.argv[1]
trainset_data = []
new_header = []
molecules = {}
header = []

# Read data in
with open(filename) as input:
    csv_reader = csv.reader(input)

    row_count = 0

    for row in csv_reader:
        row_count += 1

        # Skip header
        if row_count == 1:
            header = row
            continue

        cid = row[0]

        # If it is the first time we see that molecule, then create placeholders
        if not molecules.get(cid, ""):
            molecules[cid] = {}

            # 19 odours + pleasantness + intensity
            for i in range(21):
                molecules[cid][header[i + 1]] = []

        # Save each valid data for every odour class
        for i in range(21):
            odour_score = row[i + 1]

            # Skip empty fields
            if odour_score == "NaN":
                continue

            molecules[cid][header[i + 1]].append(int(odour_score))


# Create new header for output file
new_header.append("CID")
odours = header[-21:]
for odour in odours:
    new_header.append("MEAN_" + odour.strip())
    new_header.append("STD_" + odour.strip())

# Merge
for cid in molecules:
    mol = []
    temp = []

    # Calculate for the given molecule the mean and std value
    for k, v in molecules[cid].items():
        mean = np.mean(v)
        std = np.std(v)
        temp.append((k, mean, std))

    odours = header[-21:]

    mol.append(cid)

    # Extract calculated data in correct order
    for odour in odours:
        for i in range(21):
            field_name = temp[i][0]

            if field_name == odour:
                mean = temp[i][1]
                std = temp[i][2]

                mol.append(mean)
                mol.append(std)

    trainset_data.append(mol)


output_filename = filename[:-4]
output_filename += "Avg.csv"

with open(output_filename, "w") as output:
    csv_writer = csv.writer(output)

    csv_writer.writerow(new_header)
    csv_writer.writerows(trainset_data)
