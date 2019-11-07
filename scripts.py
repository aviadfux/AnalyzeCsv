import circles_check
import glob
import shutil
import os

#PATHS:
GRID_PATH = 'C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\generate_path\\generate\\grid_file.csv'
STUDENT_FOLDER_PATH = 'C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\generate_path\\work_students'
GENERATE_FOLDER_PATH = 'C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\generate_path\\generate'
BORDERS_SECTIONS_FOLDER_PATH = 'C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\border_sections_files'
CITIES_SECTIONS_FOLDER_PATH = 'C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\cities_sections_files'

def create_file_for_all_sections():

    file_name = "C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\border_sections.txt"
    with open(file_name, 'r') as f:
        sections = f.readline().split(',')


    dir_path = "C:\\Users\\AVIADFUX\\Desktop\\Projects\\MilitarySimulator\\data\\K_DRONE_Grid\\border_sections_files\\"
    for section in sections:
        with open(dir_path + str(section) + '.txt', 'w') as f:
            sections = f.writelines(section + ',' + section + ',')
    n=3

def create_student_paths():
    for student_path in glob.glob(STUDENT_FOLDER_PATH + '\\*'):
        student_name = student_path.split('\\')[-1]
        shutil.copy(student_path + '\\ex2.py', GENERATE_FOLDER_PATH)

        for border in glob.glob(BORDERS_SECTIONS_FOLDER_PATH + '\\*'):
            border_name = border.split('\\')[-1].split('.')[0]
            shutil.copy(border, GENERATE_FOLDER_PATH)
            os.rename(GENERATE_FOLDER_PATH + '\\' + border_name + '.txt', GENERATE_FOLDER_PATH + '\\' + 'init_cells.txt')

            for city in glob.glob(CITIES_SECTIONS_FOLDER_PATH + '\\*'):
                city_name = city.split('\\')[-1].split('.')[0]
                shutil.copy(city, GENERATE_FOLDER_PATH)
                os.rename(GENERATE_FOLDER_PATH + '\\' + city_name + '.txt', GENERATE_FOLDER_PATH + '\\' + 'goal_cells.txt')

                #get the original working directory
                owd = os.getcwd()
                #Execute student
                os.chdir(GENERATE_FOLDER_PATH)

                try:
                    os.system('python ' + 'ex2.py')
                except:
                    print(
                        "!! Generated path failed: student=[{}], border=[{}], city=[{}] !!".format(student_name,
                                                                                            border_name,
                                                                                            city_name))
                    continue
                if os.path.isfile('path.txt'):
                    path_name = border_name + '_' + city_name + '.txt'
                    os.rename('path.txt', path_name)
                    # return to original working directory
                    os.chdir(owd)


                    try:
                        check = circles_check.check(GENERATE_FOLDER_PATH + '\\' + path_name, GRID_PATH)
                    except:
                        print(
                            "Generated wrong path: student=[{}], border=[{}], city=[{}]".format(student_name,
                                                                                                border_name,
                                                                                                city_name))
                        continue

                    if not check:
                        print(
                            "Generated wrong path: student=[{}], border=[{}], city=[{}]".format(student_name,
                                                                                                          border_name,
                                                                                                          city_name))
                    else:
                        shutil.copy(GENERATE_FOLDER_PATH + '\\' + path_name, student_path + '\\' + path_name)
                        print(
                            "-- Succeeded generate path: student=[{}], border=[{}], city=[{}] --".format(student_name,
                                                                                                border_name,
                                                                                                city_name))

                    os.remove(GENERATE_FOLDER_PATH + '\\' + path_name)
                else:
                    print("Not succeeded to generate path: student=[{}], border=[{}], city=[{}]".format(student_name, border_name, city_name))
                    os.chdir(owd)

                os.remove(GENERATE_FOLDER_PATH + '\\' + 'goal_cells.txt')

            os.remove(GENERATE_FOLDER_PATH + '\\' + 'init_cells.txt')

        os.remove(GENERATE_FOLDER_PATH + '\\ex2.py')

#create_file_for_all_sections()
create_student_paths()