import csv


filename = "../../../../data/dream/TrainSet.txt"
header = None
desired_dilution = {}

intensity_data = {}
trainset_data = {}
final_trainset_data = []
cid_smiles = {}

# Read data in except intensity
with open(filename) as input:
    csv_reader = csv.reader(input, delimiter="\t")

    row_count = 0

    for row in csv_reader:
        row_count += 1

        # Skip header
        if row_count == 1:
            header = ["CID"] + row[6:]
            continue

        # Some molecules were smelled twice, skip the second smell session
        if "(replicate)" in row[1]:
            continue

        cid = row[0]

        # Not every molecule has the dilution 1/1000
        dilution = row[4].replace(",", "").replace('"', "").replace("1/", "").strip()
        if dilution == "1000":
            desired_dilution[cid] = "1000"
        elif not desired_dilution.get(cid, ""):
            desired_dilution[cid] = dilution
        elif desired_dilution[cid] and desired_dilution[cid] < dilution:
            desired_dilution[cid] = dilution

        # row[3] == Intensity
        # We only want data with a high intensity
        if row[3].strip() == "high":
            subject_id = row[5]

            if not trainset_data.get(cid, ""):
                trainset_data[cid] = {}

            # Insert dour values (except intensity) into data
            # - 1 because we don't care about intensity here
            # - 1 to skip CID
            trainset_data[cid][subject_id] = row[-(len(header) - 1 - 1) :]


# Read and insert dilution
with open(filename) as input:
    csv_reader = csv.reader(input, delimiter="\t")

    row_count = 0

    for row in csv_reader:
        row_count += 1

        # Skip header
        if row_count == 1:
            continue

        # Some molecules were smelled twice, skip the second smell session
        if "(replicate)" in row[1]:
            continue

        dilution = row[4].replace(",", "").replace('"', "").replace("1/", "").strip()

        cid = row[0]
        subject_id = row[5]

        if dilution == desired_dilution[cid]:
            scores = trainset_data[cid][subject_id]
            intensity = row[6]

            final_trainset_data.append([cid, intensity, *scores])



with open("../../../../output/old_features/data/dream/TrainSetProcessed.csv", 'w') as output:
    csv_writer = csv.writer(output)

    csv_writer.writerow(header)
    csv_writer.writerows(final_trainset_data)
