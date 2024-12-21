from libraries import *

'''
- Read the files from Food101 dataset that is already downloaded in 
    the local directory

- Note that inside the folder food-101, there is a folder called meta
    -> Inside the meta folder, there are two files called 
        classes.txt and labels.txt (We need to set these HyperParameters)
    -> There are also 2 files called test.json and train.json
        -> These files list which images are in the test and train sets

We will use the ImageFolder class from torchvision.datasets to read the
    images from the dataset
'''

train_dataframe = pd.read_json("Food101/food-101/meta/train.json")
test_dataframe = pd.read_json("Food101/food-101/meta/test.json")

# We must make 2 new folders called train and test to store the images
# We will also move the images to these folders based of the dataframes
train_dir = f"Food101/train"
test_dir = f"Food101/test"
image_dir = f"Food101/food-101/images"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

'''
Move the images to the train and test folders, 
    here is how the folders will look:
Food101
    -> train
        -> {label}
            -> {image}.jpg
    -> test
        -> {label}
            -> {image}.jpg

we need to make sure the label folders are created, then 
    check if the image is in the train or test set,
    then move the image to the correct folder
'''

if not (os.path.exists(train_dir)):

    train_count = 0
    for label, images in tqdm(train_dataframe.items()):
        # 1. Create the Label folder (The category of the image)
        label_dir = os.path.join(train_dir, label)
        if not os.path.exists(label_dir): # Always check if the folder exists
            os.makedirs(label_dir)

        # 2. Move the images to the correct folder
        for image in images:
            source_path = os.path.join(image_dir, f"{image}.jpg")
            dest_path = os.path.join(train_dir, f"{image}.jpg")
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
            train_count += 1
            
    test_count = 0
    for label, images in tqdm(test_dataframe.items()):
        label_dir = os.path.join(test_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for image in images:
            source_path = os.path.join(image_dir, f"{image}.jpg")
            dest_path = os.path.join(test_dir, f"{image}.jpg")
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
            test_count += 1
            
# print(f"Train Count: {train_count}")
# print(f"Test Count: {test_count}")

# Transforming data with torchvision.transforms.Compose
# Define the transformations
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance,
    transforms.ToTensor()
])

# Load the data using ImageFolder
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

# Data has now been loaded and transformed

'''
Now to Create a DataLoader for the training and test data
    - This is so we can load the data in batches and iterate over it
    - We shuffle the training data so the model doesn't learn the order of the data
    - Include new parameter: num_workers
        -> What is num_workers?
            -> The number of processes that generate batches in parallel.
            -> If num_workers is 0, the data will be loaded in the main process.
'''
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=NUM_OF_WORKERS)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, 
    shuffle=False, num_workers=NUM_OF_WORKERS)