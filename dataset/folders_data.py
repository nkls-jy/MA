train_dir = "./sets/train/"
test_dir = "./sets/valid/"


# create list of files for test data
import os

# test file paths
test_path = "/home/niklas/Documents/testdata_big_noisy"

# create list
#list_testfiles = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f)) and f.endswith(".tif")]

list_testfiles = [os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f)) and f.endswith(".tif")]


if __name__ == '__main__':
    print(list_testfiles)

