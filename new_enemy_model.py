import pandas as pd
import glob


def main():
    filename = "C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\sections_data_stud.csv"
    csv = pd.read_csv(filename)

    """collect all students paths"""
    students_path = []
    for filename in glob.glob('C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\general_paths\\*.txt'):
        with open(filename) as f:
            path = []
            [path.append(int(item)) for item in f.readlines()]
            students_path.append(path)

    new_df = pd.DataFrame(data=csv.values, columns=csv.columns)

    """iterate on the csv table and for all row check if needed to ascend the counter for any next node"""
    for index, row in csv.iterrows():

        try:
            section_id = row['ordinalSectionId']
            next_Sections_Id = list(map(int, row['[nextSectionsOrdinalIds]'].split('[')[1].split(']')[0].split(';')))
            next_Sections_Ids_Counters = list(map(int, row['[nextSectionsIdsCounters]'].split('[')[1].split(']')[0].split(';')))
            next_Sections_Ids_Probs = []
        except:
            continue

        if len(next_Sections_Id) == 0: continue

        """Iterate on students paths and check if exist path that goes through section_id"""
        for path in students_path:
            if section_id in path:
                """get the index of specific section and the next section, the index is from path (not table)"""
                idx_section = path.index(section_id)
                if idx_section < len(path) - 1:
                    next_section_id_from_path = path[idx_section + 1]
                    """Ascend the counter of next section from table by the ID of next section from path"""
                    if next_section_id_from_path in next_Sections_Id:
                        idx_next_section_from_table = next_Sections_Id.index(next_section_id_from_path)
                        next_Sections_Ids_Counters[idx_next_section_from_table] += 1

        """Determine new value for next_Sections_Ids_Probs based on the next sections counters"""
        sum_counters_ids = sum(next_Sections_Ids_Counters)

        if sum_counters_ids == 0:
            for idx, counter in enumerate(next_Sections_Ids_Counters): next_Sections_Ids_Counters[idx] += 1
            sum_counters_ids = sum(next_Sections_Ids_Counters)

        [next_Sections_Ids_Probs.append(float(format(section_couner/sum_counters_ids, '.6f'))) for section_couner in next_Sections_Ids_Counters]

        """Update nextSectionsIdsCounters and nextSectionsIdsProbs on new data frame table"""
        new_df.set_value(new_df.index[new_df['ordinalSectionId'] == section_id], '[nextSectionsIdsCounters]',
                         str(next_Sections_Ids_Counters).replace(',', ';').replace(' ', ''))
        new_df.set_value(new_df.index[new_df['ordinalSectionId'] == section_id], '[nextSectionsIdsProbs]',
                         str(next_Sections_Ids_Probs).replace(',', ';').replace(' ', ''))

    new_df.to_csv(path_or_buf="C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\new_pd.csv" ,index=False)
    v=1


if __name__ == '__main__':

    main()