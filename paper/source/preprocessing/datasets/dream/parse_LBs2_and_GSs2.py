import csv


# Note: The space in " CHEMICAL" is meant to be there
odours = ['INTENSITY/STRENGTH', 'VALENCE/PLEASANTNESS', 'BAKERY', 'SWEET', 'FRUIT', 'FISH', 'GARLIC', 'SPICES', 'COLD',
          'SOUR', 'BURNT', 'ACID', 'WARM', 'MUSKY', 'SWEATY', 'AMMONIA/URINOUS', 'DECAYED', 'WOOD', 'GRASS', 'FLOWER',
          ' CHEMICAL']
header = ['CID']
for odour in odours:
    header.append('MEAN_' + odour.strip())
    header.append('STD_' + odour.strip())

filenames = ['../../../data/dream/LBs2.txt', '../../../data/dream/GSs2_new.txt']

for filename in filenames:
    final_data = []
    cids = []
    molecule_data = []

    # Read data
    with open(filename) as input:
        csv_reader = csv.reader(input, delimiter='\t')

        row_count = 0

        for row in csv_reader:
            row_count += 1

            # Skip header
            if row_count == 1:
                continue

            cid = row[0]
            if cid not in cids:
                cids.append(cid)

            molecule_data.append(row)

    for cid in cids:
        mol = [cid]

        for odour in odours:
            for entry in molecule_data:
                if entry[0] != cid:
                    continue

                if entry[1] != odour:
                    continue

                mean = entry[2]
                std = entry[3]

                mol.extend([mean, std])

        final_data.append(mol)

    with open('../../../output/data/dream/' + filename.split('/')[-1][:-4] + 'Processed.csv', 'w') as output:
        csv_writer = csv.writer(output)

        csv_writer.writerow(header)
        csv_writer.writerows(final_data)
