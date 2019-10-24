import glob
import pandas as pd

def main():
    global_table = []
    for filename in glob.glob('C:\\Users\\AVIADFUX\\Desktop\\Projects\\Mor_project\\analyze\\csv file\\17-9\\*.csv'):
        csv = pd.read_csv(filename)
        if 'aviad' in filename:
            if 'a1' in filename:
                csv['exploration_algo'] = 'sp'
            else:
                csv['exploration_algo'] = 'a'
        elif 'mor' in filename:
            csv['exploration_algo'] = 'm'
        elif 'hybrid' in filename:
            csv['exploration_algo'] = 'h'

        if 'ourpths' in filename:
            csv['our_path'] = 1
        else:
            csv['our_path'] = 0

        global_table.append(csv)

    global_table = pd.concat(global_table).reset_index()
    global_table.to_csv("C:\\Users\\AVIADFUX\\Desktop\\Projects\\Mor_project\\analyze\\csv file\\17-9\\global\\global_table.csv", sep=',')

    v=8

if __name__ == '__main__':

    main()

