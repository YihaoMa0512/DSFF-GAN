import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import ResNet_seg
# from models import ResNet_seg
from tqdm import tqdm

def test(cyclegan_model_path, input_dir, output_dir, device_ids,bs):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load model and move to multiple devices using DataParallel
    G_AB =  ResNet_seg()
    G_AB = nn.DataParallel(G_AB, device_ids=device_ids)
    G_AB.to(device_ids[0])
    checkpoint = torch.load(cyclegan_model_path)
    G_AB.load_state_dict(checkpoint)
    G_AB.eval()

    # Prepare image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loop through input images and generate corresponding outputs
    batch_size = bs
    for i in tqdm(range(0, len(os.listdir(input_dir)), batch_size)):
        batch_input_images = []
        batch_input_filenames = []
        for filename in os.listdir(input_dir)[i:i+batch_size]:
            # Load input image, apply transformations, and move to device
            input_image = Image.open(os.path.join(input_dir, filename)).convert('RGB')
            input_tensor = transform(input_image)
            batch_input_images.append(input_tensor)
            batch_input_filenames.append(filename)
        input_tensor = torch.stack(batch_input_images).to(device_ids[0])

        # Generate output image and move back to CPU
        with torch.no_grad():
            output_tensor,_,_= G_AB(input_tensor)
        output_images = [(0.5*(output_tensor[j].data+1)) for j in range(len(batch_input_images))]
        print()
        # Save output images with same names as input images
        for j, filename in enumerate(batch_input_filenames):

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_image(output_images[j], os.path.join(output_dir, filename))


if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cyclegan_model_path', type=str, default='../checkpoint/dsff/epoch10/netG_A2B.pth')
    parser.add_argument('--input_dir', type=str, default='../datasets/test/A')
    parser.add_argument('--output_dir', type=str, default='../results/dsff')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--batchsize', type=int, default=32)
    args = parser.parse_args()

    test(args.cyclegan_model_path, args.input_dir, args.output_dir, args.device_ids,args.batchsize)