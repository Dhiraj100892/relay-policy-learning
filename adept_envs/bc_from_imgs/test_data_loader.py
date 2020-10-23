import torchvision.transforms as transforms
from IPython import embed
from data_loader import KitchenDatasetImg, TrainKitchenDataset

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
normal_mean_std = ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*mean_std)])

dataset = KitchenDatasetImg(traj_len=10,
            pkl_path='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data/',
            img_path='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data/imgs',
            img_transform=img_transform)

train_dataset = TrainKitchenDataset(traj_len=10,
            pkl_path='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data/',
            img_path='/private/home/dhirajgandhi/kitchen_demos_multitask_extracted_data/imgs',
            img_transform=img_transform)
        
embed()