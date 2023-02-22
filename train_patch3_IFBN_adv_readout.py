import torch
import torch.utils.data
import torch.nn as nn
import helper as h
import math
import torchvision.utils as thutil
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from setting import init_weights
import cv2
import os
from skimage import measure
import numpy as np
import glob

#from CS_recon_patch_ISTA_inco3_nor import CS_reconstruction
from CS_Net_plus_IFBN_readout import CS_reconstruction, Discriminator, Encoder, ZBottleneck

#train_data_path = r'D:/image_dataset/DIV2K/all'
#train_data_path = r'D:/image_dataset/SR_training_datasets/T91'
train_data_path = './BSD500'
#val_data_path = r'D:/image_dataset/SR_testing_datasets/Urban100'
val_data_path = './test_data'
weight_save_path = './models/CSNet+_0.01_matrix_IFBN2_n2'
cs_ratio = 0.01
max_CS_ratio = 0.1
LR_size = 32
patch_size = LR_size * 3
batchsize = 64

phase = 9
F = 64
learning_rate = 0.001
init_type = 'orthogonal'
iter = 10000


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 0.1 -> 28.88/0.8842


def image_name_load(image_path):
    file_list = os.listdir(image_path)
    return file_list

def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def patial_cubic(image):
    img = np.array(image)

    size_w = np.random.choice([0.2, 0.3], 1)[0]
    size_w = int(len(img) * size_w)

    size_h = np.random.choice([0.2, 0.3], 1)[0]
    size_h = int(len(img[0]) * size_h)

    position_x = np.random.choice(np.arange(0, len(img) - size_w, 1), 1)[0]
    position_y = np.random.choice(np.arange(0, len(img[0]) - size_h, 1))

    reduction_ratio = np.random.choice([2, 3, 4],1)[0]

    temp = img[position_x:position_x + size_w, position_y: position_y + size_h]

    fill = cv2.resize(temp, (len(temp[0])//reduction_ratio,len(temp)//reduction_ratio),interpolation=cv2.INTER_AREA)
    fill = cv2.resize(fill,(size_h,size_w),interpolation=cv2.INTER_NEAREST)

    img[position_x:position_x + size_w, position_y: position_y + size_h] = fill

    return img

def batch_img_processing(images):
    images = np.array(images)
    for i in range(len(images)):
        images[i,:,:] = patial_cubic(images[i,:,:])
    return images


def create_dataset(file_list, batchsize = 32, stride = 57, scale=4, LR_size=40, aug_time = 8, mode = 'random_crop'):
    # patch slide
    # random crop
    #patch_size = LR_size * scale
    patches = []

    if mode == 'patch_slide':
        #file_list = np.random.choice(file_list, 10)
        for i, path in enumerate(file_list):
            img = cv2.imread(train_data_path +'/'+ path, cv2.IMREAD_GRAYSCALE) /255.0
            #img = cv2.COLOR_BGR2RGB(img)
            h = len(img)
            w = len(img[0])
            for i in range(0, h - patch_size, stride):
                for j in range(0, w - patch_size, stride):
                    x = img[i:i + patch_size, j: j + patch_size]
                    for k in range(0, aug_time):
                        x_aug = data_aug(x, mode=k)
                        patches.append(x_aug)
        # np.random.shuffle(patches)
        # patches = patches[0:batchsize]
        print(np.shape(patches))
    elif mode == 'random_crop':
        file_list = np.random.choice(file_list,batchsize)
        for i, path in enumerate(file_list):
            img = cv2.imread(train_data_path +'/'+ path, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]/ 255.0

            h = len(img)
            w = len(img[0])
            if h <= patch_size:
                h_ = 0
            else:
                h_ = np.random.choice(h - patch_size, 1)[0]
            if w <= patch_size:
                w_ = 0
            else:
                w_ = np.random.choice(w - patch_size, 1)[0]

            x = img[h_:h_ + patch_size, w_: w_ + patch_size]
            for k in range(0, aug_time):
                x_aug = data_aug(x, mode=np.random.randint(0, 8))
                patches.append(x_aug)
    else :
        print("mode incorrect!")

    return patches
def test(model):
    val_list = image_name_load(val_data_path)
    PSNR = []
    SSIM = []
    model.eval()

    for i, CS in enumerate(CS_ratio):
        p = []
        s = []
        for i, name in enumerate(val_list):
            img = cv2.imread(val_data_path + '/' + name, cv2.IMREAD_GRAYSCALE)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            h = len(img) // LR_size
            w = len(img[0]) // LR_size

            img = img[0:h*LR_size, 0: w * LR_size] / 255.0
            input_val = ToTensor(np.reshape(img,[1, h*LR_size,w*LR_size,1])).float().cuda()
            recon, _, _ = model(input_val, CS, 0)
            output_val = np.reshape(recon.cpu().detach().numpy()[:,0,:,:],[h*LR_size, w*LR_size]) * 255

            p.append(measure.compare_psnr(img * 255,output_val,data_range=255))
            s.append(measure.compare_ssim(img* 255, output_val, data_range=255))

        PSNR.append(np.mean(p))
        SSIM.append(np.mean(s))

    return PSNR, SSIM

def ToTensor(images):
    # numpy image: B X H x W x C
    # torch image: B X C X H X W
    if len(np.shape(images)) == 4:
        images = np.array(images).transpose((0, 3, 1, 2))
    else :
        b,h,w = np.shape(images)
        images = np.reshape(images,[b,h,w,1])
        images = np.array(images).transpose((0, 3, 1, 2))
    return torch.from_numpy(images)

img_pathes = image_name_load(train_data_path)
#train_images = create_dataset(img_pathes,mode='random_crop')
train_images = create_dataset(img_pathes, mode='patch_slide')
#tensor_images = ToTensor(train_images)

# network definition
discriminator = Discriminator(LR_size,max_CS_ratio)
discriminator = nn.DataParallel(discriminator).cuda()
init_weights(discriminator, 'kaiming')

zBottleneck = ZBottleneck(LR_size,max_CS_ratio)
zBottleneck = nn.DataParallel(zBottleneck).cuda()
init_weights(zBottleneck, 'kaiming')

generator = Encoder(LR_size,max_CS_ratio)
generator = nn.DataParallel(generator).cuda()
init_weights(generator, 'kaiming')

model = CS_reconstruction(F,cs_ratio, LR_size ,phase=phase, G=generator, Z=zBottleneck, max_CS_ratio=max_CS_ratio)
model = nn.DataParallel(model).cuda() # gpu 사용
init_weights(model, init_type)
print('the number of parameters : %d '%sum(p.numel() for p in model.parameters() if p.requires_grad))



# loss and optimizer
criterion = nn.MSELoss().cuda()
criterion1 = nn.L1Loss().cuda()
criterion2 = nn.BCELoss().cuda()

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
optimizer_G = optim.Adam(generator.parameters(),lr=learning_rate)
optimizer_Z = optim.Adam(zBottleneck.parameters(),lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(),lr=learning_rate)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma= 0.1)
scheduler = lr_scheduler.StepLR(optimizer, step_size=25,gamma=0.5)
scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=25,gamma=0.5)
scheduler_Z = lr_scheduler.StepLR(optimizer_Z, step_size=25,gamma=0.5)
scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=25,gamma=0.5)

#load last check point
#file_list = os.listdir(weight_save_path)
file_list = glob.glob(weight_save_path + '/*.pth')
file_list.sort()
if file_list == []:
    start_point = 0
    print("Train from initial")
else :
    w_name = file_list[-1]
    check_p = torch.load(w_name)
    model.load_state_dict(check_p['mainmodel'])
    generator.load_state_dict(check_p['G'])
    discriminator.load_state_dict(check_p['D'])
    zBottleneck.load_state_dict(check_p['Z'])
    #optimizer.load_state_dict(check_p['optimizer_state_dict'])
    start_point = len(file_list)
    print("Train from last check point %s"%file_list[-1])

output_file_name = weight_save_path + "/log.txt"

CS_ratio = [0.1, 0.1, 0.1, 0.1]
CS_mode = CS_ratio[0]
p_temp = [0]
IFBN_loss = torch.tensor(0)
E_IFBN_loss = torch.tensor(0)
# train1
for epoch in range(start_point, iter):
    running_loss = 0
    running_main = 0
    running_G = 0
    running_D = 0
    running_rip_H = 0
    running_y = 0
    running_E = 0
    fake_judge = np.array([])
    real_judge = np.array([])

    scheduler.step(epoch)
    scheduler_D.step(epoch)
    scheduler_G.step(epoch)
    scheduler_Z.step(epoch)
    model.train()


    np.random.shuffle(train_images)
    for i in range(1400):
        # dataset
        #train_images = create_dataset(img_pathes, batchsize=batchsize)
        #np.random.shuffle(train_images)
        label = train_images[i*batchsize:(i+1)*batchsize]
        #train_imgs = batch_img_processing(label)
        train_imgs = label #+ np.random.standard_normal(np.shape(label)) / 255.0
        #IFBN_imgs = cv2.GaussianBlur(np.array(label),(9,9),0)
        tensor_images = ToTensor(train_imgs)
        label_images = ToTensor(label)
        #IFBN_images = ToTensor(IFBN_imgs)
        data = tensor_images.float()
        label_data = label_images.float()
        #IFBN_tensor = IFBN_images.float()
        # select CS mode

        CS_mode = np.random.choice(CS_ratio[0:1], 1)[0]

        # estimate
        C = int(LR_size * LR_size * CS_mode)
        LR = data.view(batchsize, 1, patch_size, patch_size).cuda()
        HR = label_data.view(batchsize,1,patch_size,patch_size).cuda()
        #IR = IFBN_tensor.view(batchsize, 1, patch_size, patch_size).cuda()


        ######################################### IFBN train #####################################
        beta = 1e-5
        lamb = 1e-5




        result, _, _ = model(LR, CS_mode, 1) # 1 means using IFBN z

        Z_main_loss = criterion(result, HR)
        BN_loss = zBottleneck.module.buffer_capacity.mean()

        IFBN_loss = Z_main_loss + beta * BN_loss
        IFBN_loss = IFBN_loss / (2*batchsize)


        # IFBN learning
        optimizer_Z.zero_grad()
        IFBN_loss.backward()
        optimizer_Z.step()


        # Encoder learning
        _, _, _ = model(LR, CS_mode, 1)  # 1 means using IFBN z

        E_IFBN_loss = criterion(zBottleneck.module.lamb,
                                torch.ones_like(zBottleneck.module.lamb, device='cuda:0'))
        Encoder_loss = lamb * E_IFBN_loss / (batchsize)

        optimizer_G.zero_grad()
        Encoder_loss.backward()
        optimizer_G.step()


        ######################################### MAIN train ######################################
        result, y, rip_H = model(LR, CS_mode, 0)

        main_loss = criterion(result, HR)
        rip_H_loss = criterion(rip_H, HR)

        r1 = 0.1
        Total_loss = main_loss + r1 * rip_H_loss
        Total_loss = Total_loss / (2*batchsize)
        # gradient zero
        optimizer.zero_grad()
        # back propagation
        Total_loss.backward()
        optimizer.step()




        running_loss += Total_loss.item()
        running_main += main_loss.item()
        running_y += IFBN_loss.item()
        running_E += E_IFBN_loss.item()
        running_rip_H += rip_H_loss.item()


    # end epoch
    print('[epoch : %d] loss : %f, main_loss : %f, rip_H_loss : %f, IFBN_loss : %f, Encoder_train : %f'%(epoch, running_loss, running_main, running_rip_H, running_y, running_E))
    #print("Learning Rate: {}".format(scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
    print(zBottleneck.module.lamb.mean().item())

    torch.save({'mainmodel':model.state_dict(),
                'G':generator.state_dict(),
                'D':discriminator.state_dict(),
                'Z':zBottleneck.state_dict()}, weight_save_path + '/%04d.pth'%epoch)
    with torch.no_grad():
        psnr, ssim = test(model)
        p_temp = psnr
        print(' 0.01 : %.2f, 0.1 : %.2f, 0.3 : %.2f, 0.4 : %.2f'%(psnr[0], psnr[1],psnr[2],psnr[3]))
    output_data = '[epoch : %d] loss : %f, main_loss : %f, rip_H_loss : %f, G_loss : %f, Encoder_train : %f, 0.01 : PSNR : %.2f, SSIM : %.4f, 0.1 : PSNR : %.2f, SSIM : %.4f, 0.3 : PSNR : %.2f, SSIM : %.4f' \
                  ',0.4 : PSNR : %.2f, SSIM : %.4f\n' % (
        epoch, running_loss, running_main, running_rip_H, running_y, running_E, psnr[0], ssim[0],psnr[1],ssim[1],psnr[2], ssim[2], psnr[3],ssim[3])
    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()
