import os
from glob import glob
import natsort
from optparse import OptionParser


def remove_checkpoints(search_dir):

    for filename in os.listdir(search_dir):
        dir_checkpoint = os.path.join(search_dir,filename)

        if os.path.isdir(dir_checkpoint) == True:
            checkpoints = natsort.natsorted(glob(os.path.join(dir_checkpoint, "*.pth")))

            # remove all checkpoints but the last 
            checkpoints_to_remove = checkpoints[:-1] 
            if len(checkpoints_to_remove) > 0:
                for checkpoint in checkpoints_to_remove:
                    os.remove(checkpoint)


def get_args():
          
    parser = OptionParser()
    parser.add_option('-d', '--dir', dest='dir', default='../checkpoints', help='checkpoints directory')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    
    args = get_args()
    print('Removing all checkpoints but the last one in ', args.dir)
    remove_checkpoints(args.dir) #