import sys
import pandas as pd

def neighbors_section_check(section_list, grid_file):
    #grid_file = "C:/Users/AVIADFUX/Desktop/Projects/MilitarySimulator/DroneSim/data/sections_data_.csv"
    sections_data = pd.read_csv(grid_file, error_bad_lines=False)

    last_section = int(section_list[0])
    for section in section_list[1:]:
        if section == '': continue
        current_section = int(section)
        neighbors_list = list(map(int, sections_data.iloc[int(last_section)]['[nextSectionsOrdinalIds]']
                                  .replace('[', "").replace(']', "").split(';')))

        if current_section not in neighbors_list:
            #print("### BAD neighbors_section_check###")
            return False

        last_section = current_section

    return True

def check(path, grid_path):
    with open(path) as file:
        sections_list = file.read().splitlines()

    if not neighbors_section_check(sections_list, grid_path): return False

    sections_set = set()

    for section in sections_list:
        if section in sections_set:
            #print("### BAD circles_check###")
            return False

    return True
    #print("### GOOD ###")


def main():
    grid_path = "C:/Users/AVIADFUX/Desktop/Projects/MilitarySimulator/DroneSim/data/sections_data_.csv"
    check(sys.argv[1], grid_path)

if __name__ == '__main__':
    main()