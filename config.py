base_dir_sls = "data/base_sls"
target_sls = "data/sls"
test_sls = "tests/sls"
base_dir_nosls = "data/base_nosls"
target_nosls = "data/nosls"
test_nosls = "tests/nosls"
random_dir = "data/random_images"
test_random = "tests/random_images"
nb_image_sls = 10000
nb_image_nosls = 5000
nb_test_sls = 1000
nb_test_nosls = 500

batch_size = 100
epochs = 30
lr = 0.01
momentum = 0.9
val_size = 0.1
img_size = 42
model_path = "model/sls_classifier.pth"