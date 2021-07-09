import torch
from torchvision.transforms import transforms, InterpolationMode
from torchvision import datasets
from torchvision.datasets import ImageFolder



class ModelReadyDataCreator:

  def __init__(self, train_path, valid_path, test_path, project_path, data_path, model_weights_path, height=512, width=512, batch_size=4):
    self.height = height
    self.width = width
    self.project_path = project_path
    self.data_path = data_path
    self.model_weights_path = model_weights_path
    self.phase_list = ['train','valid','test']
    self.dirs = {self.phase_list[0] : train_path, 
                self.phase_list[1] : valid_path, 
                self.phase_list[2] : test_path}
    self.batch_size = batch_size

    self.image_datasets = {x : ImageFolder(self.dirs[x], transform=self.get_transforms()) for x in self.phase_list}
    self.data_loaders = {x : torch.utils.data.DataLoader(self.image_datasets[x], batch_size = self.batch_size, shuffle = True, num_workers = 2) for x in self.phase_list}
    self.dataset_sizes = {x : len(self.image_datasets[x]) for x in self.phase_list}
    self.class_names = self.image_datasets[self.phase_list[0]].classes
    self.num_of_classes = len(self.class_names)
    self.class_to_idx = self.image_datasets['train'].class_to_idx
    self.idx_to_class = {val : key for key, val in self.class_to_idx.items()}

  def get_model_weight_paths(self):
    return self.model_weights_path

  def get_class_to_idx(self):
    return self.class_to_idx

  def get_idx_to_class(self):
    return self.idx_to_class

  def get_number_of_classes(self):
    return self.num_of_classes
    
  def get_dataset_sizes(self):
    return self.dataset_sizes
  
  def get_class_names(self):
    return self.class_names

  def get_image_datasets(self): 
    return self.image_datasets

  def get_data_loaders(self):
    return self.data_loaders

  def get_dirs(self):
    return self.dirs

  def get_dimension(self):
    return self.height, self.width

  def get_project_path(self):
    return self.project_path

  def get_data_path(self):
    return self.data_path

  def get_transforms(self, grayscale=False, interpolation_method=InterpolationMode.NEAREST):
    """
    function to return transforms object with all the transformations
    param 0 : resize height
    param 1 : resize width
    param 2 : grayscale image flag
    param 3 : Interpolation method used while resizing
    """

    transformation_list = []

    if grayscale : 
      transformation_list.append(transforms.Grayscale(1))

    transformation_list.append(transforms.Resize((self.height,self.width), interpolation=interpolation_method))
    transformation_list.append(transforms.ToTensor())

    if grayscale :
      transformation_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
      transformation_list.append(transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))

    return transforms.Compose(transformation_list)
