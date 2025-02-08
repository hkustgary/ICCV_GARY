import dsmil as mil
import openslide as slide
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings

sys.path.append('../')
from score_prolif import prolif
from extract_features import get_patch_uni_feature

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray([int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])]) # row, col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

# def patch_and_feature():
#     bags_list = glob.glob(os.path.join('test-c16', 'patches', '*'))
#     feats_list = []
#     pos_list = []
#     classes_list = []
#     print(bags_list)
#     csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
#     dataloader, bag_size = bag_dataset(args, csv_file_path)
#     for iteration, batch in enumerate(dataloader):
#             patches = batch['input'].float().cuda()
#             patch_pos = batch['position']
#             feats, classes = milnet.i_classifier(patches)
#             feats = feats.cpu().numpy()
#             classes = classes.cpu().numpy()
#             feats_list.extend(feats)
#             pos_list.extend(patch_pos)
#             classes_list.extend(classes)
#     pos_arr = np.vstack(pos_list)
#     feats_arr = np.vstack(feats_list)
#     return pos_arr, feats_arr
def pre_prolif(pred_matrix, coords, features, threshold,feature_threshold, dense_threshold,corr = False):
    patch_size = 224
    # non_zero_elements = pred_matrix[pred_matrix > 0]
    # num_elements = int(threshold * len(non_zero_elements))
    # new_thresh = np.partition(non_zero_elements, num_elements - 1)[num_elements - 1] if num_elements > 0 else 0

    cancer_flag = np.full(len(coords), False)
    scores = np.zeros(len(coords))
    for idx,coord in enumerate(coords):
        cancer_flag[idx] = pred_matrix[coord[0], coord[1]]>threshold
        scores[idx] = pred_matrix[coord[0], coord[1]]
    new_coords = coords * patch_size
    new_scores = prolif(scores,new_coords, cancer_flag, features, feature_threshold, dense_threshold, patch_size,corr)
    return new_scores

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores/100
def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        slide_name = bags_list[i].split(os.sep)[-1]
        print('Processing ' + slide_name)
        if os.path.exists('../../data/predict/dsmil/ori/'+slide_name+"_predict.npy"):
            print(slide_name)
            exist_eval_res(slide_name)
            continue

        features_list = []
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        feature_list_path = 'pred/'+slide_name+'/'+slide_name+"_features.npy"
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                if not os.path.exists(feature_list_path):
                    features = get_patch_uni_feature(patches)
                    features_list.extend(features)
                else:
                    features_list = np.load(feature_list_path)
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            os.makedirs('pred/'+bags_list[i].split(os.sep)[-1], exist_ok=True)
            np.save('pred/'+bags_list[i].split(os.sep)[-1]+'/'+bags_list[i].split(os.sep)[-1]+"_feat.npy", feats_arr)
            np.save('pred/'+bags_list[i].split(os.sep)[-1]+'/'+bags_list[i].split(os.sep)[-1]+"_coor.npy", pos_arr)
            np.save(feature_list_path, features_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            color = [0, 0, 0]
            if bag_prediction >= args.thres_tumor:
                print(bags_list[i] + ' is detected as malignant')
                color = [1, 0, 0]
                attentions = A
            else:
                attentions = A
                print(bags_list[i] + ' is detected as benign')

            slide_name = bags_list[i].split(os.sep)[-1]
            # wsi_path = os.path.join('/home/mdi/WS-FSS-Code/baselines/dsmil-wsi/test-c16/input/', slide_name + '.tif')
            wsi_path = os.path.join('/home/mdi/WS-FSS-Code/data/wsi/test/', slide_name + '.tif')

            img = slide.OpenSlide(wsi_path)
            dimension = img.level_dimensions[1] # given as width, height
            step_y_max = int(np.floor(dimension[1]/224)) # rows
            step_x_max = int(np.floor(dimension[0]/224)) # columns


            color_map = np.zeros((step_y_max, step_x_max, 3))
            patch_map = np.zeros((step_y_max, step_x_max))
            # color_map = np.zeros((np.amax(pos_arr, 0)[0], np.amax(pos_arr, 0)[1], 3))
            # patch_map = np.zeros((np.amax(pos_arr, 0)[0], np.amax(pos_arr, 0)[1]), dtype=np.uint8)
            attentions = attentions.cpu().numpy()
            attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
            scores = to_percentiles(attentions)
            for k, pos in enumerate(pos_arr):
                try:
                    tile_color = np.asarray(color) * attentions[k]
                    color_map[pos[0], pos[1]] = tile_color
                    patch_map[pos[0],pos[1]] = scores[k]
                except Exception as e:
                    print(e)
                    continue

            eval_res(features_list, slide_name, pos_arr, step_y_max, step_x_max, color_map, patch_map) 

def exist_eval_res(slide_name):

            # wsi_path = os.path.join('/home/mdi/WS-FSS-Code/baselines/dsmil-wsi/test-c16/input/', slide_name + '.tif')
    wsi_path = os.path.join('/home/mdi/WS-FSS-Code/data/wsi/test/', slide_name + '.tif')

    img = slide.OpenSlide(wsi_path)
    dimension = img.level_dimensions[1] # given as width, height
    step_y_max = int(np.floor(dimension[1]/224)) # rows
    step_x_max = int(np.floor(dimension[0]/224)) # columns
    features_list = np.load('pred/'+slide_name+'/'+slide_name+"_features.npy")
    pos_arr = np.load('pred/'+slide_name+'/'+slide_name+"_coor.npy")
    print(slide_name)
    patch_map = np.load('../../data/predict/dsmil/ori/'+slide_name+"_predict.npy")
    eval_res(features_list, slide_name, pos_arr, step_y_max, step_x_max, None, patch_map)
def eval_res(features_list, slide_name, pos_arr, step_y_max, step_x_max, color_map, patch_map):
    thresh = 0.90

    prolif_res = pre_prolif(patch_map, pos_arr, features_list, thresh, 0.7, 3)
    corr_res = pre_prolif(patch_map, pos_arr, features_list, thresh, 0.7, 3, True)
            
    new_prolif_res = np.zeros((step_y_max, step_x_max))
    new_corr_prolif_res = np.zeros((step_y_max, step_x_max))
    for k, pos in enumerate(pos_arr):
        try:
            if prolif_res[k]:
                new_prolif_res[pos[0],pos[1]] = 1
            else:
                new_prolif_res[pos[0],pos[1]] = 0
            if corr_res[k]:
                new_corr_prolif_res[pos[0],pos[1]] = 1
            else:
                new_corr_prolif_res[pos[0],pos[1]] = 0
        except Exception as e:
            continue
            
            ##################
    
    patch_map = np.where(patch_map > thresh, 1, 0)
    np.save('../../data/predict/dsmil/ori/'+slide_name+"_predict.npy", patch_map)            
    np.save('../../data/predict/dsmil/prolif/'+slide_name+"_predict.npy", new_prolif_res)
    np.save('../../data/predict/dsmil/prolif_cor/'+slide_name+"_predict.npy", new_corr_prolif_res)
    if color_map is not None:
        color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
        io.imsave(os.path.join('test-c16', 'output', slide_name+'.png'), img_as_ubyte(color_map), check_contrast=False)       
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.5282700061798096)
    args = parser.parse_args()
    resnet = models.resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    # milnet.load_state_dict(torch.load('test/mil_weights_fold_4.pth'), strict=False)
    aggregator_weights = torch.load('example_aggregator_weights/c16_aggregator.pth')
    milnet.load_state_dict(aggregator_weights, strict=False)
    
    state_dict_weights = torch.load(os.path.join('test-c16', 'weights', 'embedder.pth'))
    new_state_dict = OrderedDict()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    new_state_dict["fc.weight"] = aggregator_weights["i_classifier.fc.0.weight"]
    new_state_dict["fc.bias"] = aggregator_weights["i_classifier.fc.0.bias"]
    i_classifier.load_state_dict(new_state_dict, strict=True)
    milnet.i_classifier = i_classifier
    
    bags_list = glob.glob(os.path.join('test-c16', 'patches', '*'))
    # bags_list = glob.glob(os.path.join('test-c16', 'test_patches', '*'))

    os.makedirs(os.path.join('test-c16', 'output'), exist_ok=True)
    test(args, bags_list, milnet)