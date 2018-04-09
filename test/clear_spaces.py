import os


if __name__ == '__main__':
    workdir = '/home/jullygh/Dataset_riie/All'

    for root, dirs, files in os.walk(workdir):
        for dir in dirs:
            if 'espacio_' in dir:
                dir_to_delete = os.path.join(root, dir)
                print(dir_to_delete)
