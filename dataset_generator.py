import config as cfg
import cv2
import os

from image_processing import random_transformations


def create_directories():
    print("################## Creation of directories #################")
    if not (os.path.exists(cfg.target_sls)):
        os.mkdir(cfg.target_sls)
    if not (os.path.exists(cfg.target_nosls)):
        os.mkdir(cfg.target_nosls)
    if not (os.path.exists(cfg.test_nosls)):
        os.mkdir(cfg.test_nosls)
    if not (os.path.exists(cfg.test_sls)):
        os.mkdir(cfg.test_sls)

def generate(img_folder, destination_folder, num_imgs):
    print(f"################## Generating images for {img_folder} #################")
    for image_file in os.listdir(img_folder):
        image_path = os.path.join(img_folder, image_file)
        img = cv2.imread(image_path)
        for i in range(num_imgs//len(os.listdir(img_folder))):
            generated_img = random_transformations(img)
            generated_img_path = os.path.join(destination_folder, image_file.split(".")[
                                    0] + "_" + str(i) + ".jpg")
            cv2.imwrite(generated_img_path, generated_img)
    print(f"################## {num_imgs} images generated #################")

def generate_dataset():
    create_directories()
    generate(cfg.base_dir_sls, cfg.target_sls, cfg.nb_image_sls)
    generate(cfg.base_dir_nosls, cfg.target_nosls, cfg.nb_image_nosls)
    generate(cfg.base_dir_sls, cfg.test_sls, cfg.nb_test_sls)
    generate(cfg.base_dir_nosls, cfg.test_nosls, cfg.nb_test_nosls)

if __name__ == "__main__":
    generate_dataset()