import os
import shutil

def delete_content_inside_directory(directoryPath):
    '''

    :param directoryPath:
    :return:
    '''
    for root, dirs, files in os.walk(directoryPath):
        print('clean directory {0}'.format(dirs))
        for f in files:
            print('drop file {0}'.format(f))
            os.unlink(os.path.join(root, f))
        for d in dirs:
            print('drop directory {0}'.format(d))
            shutil.rmtree(os.path.join(root, d))

delete_content_inside_directory('./data/input/')
delete_content_inside_directory('./data/train/')
delete_content_inside_directory('./data/output/')
delete_content_inside_directory('./model')


