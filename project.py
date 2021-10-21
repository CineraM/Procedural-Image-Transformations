'''
Project 1 Transformations        
Matias Cinera 
U 6931-8506     
01/28/21
'''
import imageio, sys, os, numpy as np, zipfile, math
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

#collab try block
try:
  from google.colab import drive
  drive_dir = '/content/drive'
  drive.mount(drive_dir, force_remount=True)

# output directory
  output_dir = '/content/drive/MyDrive/Colab Notebooks/computer_vision_fa21/project-1/output'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  import gdown
  data_dir_parent = '/content'
  data_dir = os.path.join(data_dir_parent, 'data')
  data_zip = '/content/data.zip'
  url = 'https://drive.google.com/uc?id=1q0OHxybXOz9bNmNf4s4wg8P65KF03WZe'
  gdown.download(url, data_zip, quiet=False)
  
  with zipfile.ZipFile(data_zip, 'r') as zf:
        zf.extractall(data_dir_parent)
        print(f'\nlist of {data_dir} directory: {os.listdir(data_dir)}\n')
  
  COLAB = Ture

except Exception as e:
    COLAB = False
    data_dir = None
    output_dir = None

# Helper Functions From the TA

def degree_to_radian(degree):
    return degree * np.pi / 180.0

# Create output filename and path
def get_output_path(output_dir, img_path, trans):
    # Captures full fname from path and splits at fname and extension
    img_fname, img_ext = os.path.split(img_path)[-1].rsplit('.', 1)

    # Creates new fname with extension and adds full output path
    new_fname_ext = f'{img_fname}-{trans}.{img_ext}'
    out_path = os.path.join(output_dir, new_fname_ext)

    return out_path

# Gets paths to images in data directory
def get_image_paths(data_dir):
    # Pull image files in dir, must be png, jpg, jpeg
    img_exts = ['png', 'jpg', 'jpeg']
    img_list = os.listdir(data_dir)
    img_list = [f for f in img_list if f.rsplit('.', 1)[-1] in img_exts]

    # Check to make sure it found images, exit if not
    if not img_list:
        print(f'No images found in {data_dir}. Exiting program...\n') 
        sys.exit(1)

    # Add directory to path of image files
    img_path_list = [os.path.join(data_dir, f) for f in img_list]
    img_path_list = [f for f in img_path_list if os.path.isfile(f)]
    img_path_list.sort()

    # Check to make sure it found image paths, exit if not
    if not img_path_list:
        print(f'No images found in {data_dir}. Exiting program...\n')
        sys.exit(1)

    return img_path_list
##### MY HELPERS END #####

##### TRANSFORMATION CLASS START #####
# Tranformation Class with all transformation functions
class transformations:
    # Image setup function
    def __setup_image(self, img_path):
        # Open Image
        self.img = imageio.imread(img_path)

        # Set width and height of original image
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

        # Check if rgb or grayscale
        if self.img.ndim < 3:
            self.isgray = True
            self.isrgb  = False
            self.img = self.img.reshape((self.height, self.width, 1))
        elif self.img.shape[2] == 1:
            self.isgray = True
            self.isrgb  = False
        else:
            self.isgray = False
            self.isrgb  = True

        # Check for alpha channel and remove if exists
        if self.isrgb and self.img.shape[2] > 3:
            self.img = self.img[..., :3]

        # Get number of channels, 1 is grayscale, 3 is RGB
        self.channels = self.img.shape[2]

        # Shape tuple
        self.shape = self.img.shape 

    # Get batch array function
    def __get_batch_array_list(self, batch_paths):
        # Goes through images and adds to batch list
        batch_paths.sort()
        batch_img_list = []
        for img_path in batch_paths:
            self.__setup_image(img_path)
            batch_img_list.append(self.img.copy())

        # Saves batch
        self.batch = batch_img_list

    # Constructor
    def __init__(self, img_path, batch_paths=None):
        # Sets up image
        if img_path:
            self.__setup_image(img_path)

        # Batch array list
        if batch_paths:
            self.__get_batch_array_list(batch_paths)


    ### PROVIDED FUNCTIONS ###
    # Inverse Warp A
    def inverse_warp_a(self, h, h_inv):
        # Get 4 corner points
        cx, cy = [], []
        for fx in [0, self.width - 1]:
            for fy in [0, self.height - 1]:
                x, y = h(fx, fy)
                x, y = int(x), int(y)
                cx.append(x)
                cy.append(y)

        # Get min and max, then new width and height
        min_x, max_x = int(min(cx)), int(max(cx))
        min_y, max_y = int(min(cy)), int(max(cy))
        width_g = max_x - min_x + 1
        height_g = max_y - min_y + 1

        # Creates empty new image
        img_new = np.zeros((height_g, width_g, self.channels))

        # Find pixel values and map to new image
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                fx, fy = h_inv(gx, gy)
                fx, fy = int(fx), int(fy)
                if fx in range(self.width) and fy in range(self.height):
                  img_new[gy - min_y, gx - min_x] = self.img[fy, fx]

        # Returns new image
        return img_new

    # Inverse Warp B
    def inverse_warp_b(self, h_inv, output_shape):
        # Create empty new image
        if len(output_shape) < 3:
            output_shape = output_shape + (self.channels,)
        img_new = np.zeros(output_shape)

        # Find pixel values and map to new image
        for gy in range(output_shape[0]):
            for gx in range(output_shape[1]):
                fx, fy = h_inv(gx, gy)
                fx, fy = int(fx), int(fy)
                if fx in range(self.width) and fy in range(self.height):
                  img_new[max(0, gy), max(0, gx)] = self.img[fy, fx]

        # Returns new image
        return img_new

    ### HELPERS END ###


    ### EXAMPLES START ###
    # Flip vertical
    def flip_vertical(self):
        # h and h inverse using nested functions
        def h(x, y):
            gx = x
            gy = abs(y - (self.height - 1))
            return (gx, gy)
        def h_inv(x, y):
            fx = x
            fy = abs(y - (self.height - 1))
            return (fx, fy)

        # Warp image
        img_new = self.inverse_warp_a(h, h_inv)
        # img_new = self.inverse_warp_b(h_inv, self.shape)

        # Return new image
        return img_new.astype(np.uint8)

    # Flop Horizontal
    def flip_horizontal(self):
        # h and h inverse using lambda functions instead
        h = lambda x, y: (abs(x - (self.width - 1)), y)
        h_inv = lambda x, y: (abs(x - (self.width - 1)), y)

        # Warp image
        # img_new = self.inverse_warp_a(h, h_inv)
        img_new = self.inverse_warp_b(h_inv, self.shape)

        # Return new image
        return img_new.astype(np.uint8)

    ### END ###

    ### Personal Implementation ###
    def translation(self, shift_x, shift_y):
        height, width, dummy = self.shape
        output_shape = self.shape
        sx, sy = width*shift_x, height*shift_y
        img_new = np.zeros(self.shape, dtype=np.uint8)

        for gx in range (output_shape[1]):
          for gy in range (output_shape[0]):
            fx = int(gx + sx)
            fy = int(gy + sy)
            if(0 <= fy < height):       # check if index is out of range
              if(0 <= fx < width):
                img_new[fy][fx] = self.img[gy][gx]
        
        # tranform the img before returning
        return img_new.astype(np.uint8)


    def rotate(self, theta):
      img_new = np.zeros(self.shape)
      theta = -theta    #inverting theta
      def h (x, y):
        R = np.array([
            [np.cos(theta), - np.sin(theta), 1],
            [np.sin(theta),   np.cos(theta), 1],
            [0            , 0              ,  1 ]])
        
        out = R @ np.array([[x],[y],[1]])
        return (out[0]/out[2], out[1]/out[2])

      def h_inv (x, y):
        R = np.array([
            [np.cos(theta),    np.sin(theta), -1],
            [-np.sin(theta),   np.cos(theta), -1],
            [0            , 0              ,  1 ]])
        
        out = R @ np.array([[x],[y],[1]])
        return (out[0]/out[2], out[1]/out[2])

      img_new = self.inverse_warp_a(h, h_inv)
      return img_new.astype(np.uint8)



    def scale(self, scale_percent):
      s = scale_percent
      img_new = np.zeros(self.shape)
      theta = 0
      def h (x, y):
        R = np.array([  #fixed transpose
            [s*np.cos(theta), - s*np.sin(theta), 1],
            [s*np.sin(theta),   s*np.cos(theta), 1],
            [0            , 0              ,  1 ]])
        
        out = R @ np.array([[x],[y],[1]])
        return (out[0]/out[2], out[1]/out[2])

      def h_inv (x, y):
        R = np.array([   #fixed transpose
            [np.cos(theta)/s,    np.sin(theta)/s, -1],
            [-np.sin(theta)/s,   np.cos(theta)/s, -1],
            [0            , 0              ,  1 ]])
        
        out = R @ np.array([[x],[y],[1]])
        return (out[0]/out[2], out[1]/out[2])

      img_new = self.inverse_warp_a(h, h_inv)
      
      return img_new.astype(np.uint8)
      
      

    def affine(self, A):
      img_new = np.zeros(self.shape)
      new_A = np.array(A)
      x, y = new_A.shape
      a = np.ones([x,y])
      square_a = np.ones([x,y-1])
      
      for i in range(x):    #Copy the A vector
        for j in range(y):
          a[i][j] = A[i][j]
      
      for i in range(len(square_a)):#Numpy arr for compatibility
        for j in range(len(square_a[0]-1)):
          square_a[i][j] =  new_A[i][j]       


      def h (x, y):
        R = np.array([
            [a[0,0], a[0,1], a[0, 2]],
            [a[1,0], a[1,1], a[1, 2]],
            [ 0 , 0 , 1 ]])
        
        out = R @ np.array([[x],[y],[1]])
        return (out[0]/out[2], out[1]/out[2])

      def h_inv (x, y):
        b = np.linalg.inv(square_a)
        R = np.array([
            [b[0,0], b[0,1], -a[0, 2]],
            [b[1,0], b[1,1], -a[1, 2]],
            [ 0 , 0 , 1 ]])
        
        out = R @ np.array([[x],[y],[1]])
        return (out[0]/out[2], out[1]/out[2])

      img_new = self.inverse_warp_a(h, h_inv)
      
      return img_new.astype(np.uint8)



    def projective(self, H):
        img_new = np.zeros(self.shape)
        new_h = np.array(H) #Numpy arr for compatibility
                
        def h (x, y):
          out = new_h @ np.array([[x],[y],[1]])
          return (out[0]/out[2], out[1]/out[2])

        def h_inv (x, y):
          H_inv = np.linalg.inv(new_h)
          out = H_inv @ np.array([[x],[y],[1]])
          return (out[0]/out[2], out[1]/out[2])

        img_new = self.inverse_warp_a(h, h_inv)          
        return img_new.astype(np.uint8)



    def brightness_contrast(self, a, b): 

      def change_contrast(input_im, new_a):
        out_im = out_im = new_a*input_im
        out_im = np.where(out_im > 1, 1, out_im) # clip the intensity to max 1 
        out_im = np.where(out_im < 0, 0, out_im) # clip the intensity to min 0
        return out_im

      def change_brightness(input_im, new_b):
        out_im = input_im + new_b
        out_im = np.where(out_im > 1, 1, out_im) # clip the intensity to max 1 
        out_im = np.where(out_im < 0, 0, out_im) # clip the intensity to min 0
        return out_im
      
      original = self.img
      original = original.astype(np.float)
      original /= np.max(original)

      trans = rgb2lab(original)
      trans[:,:,0] = change_contrast(change_brightness(trans[:,:,0]/100, b), a) * 100.0
      # *100 so the range is between 0 and 255 
      # return as int (image)
      return trans[:,:,0].astype(np.uint8)


    def gamma_correction(self, a, b):

      def gc (input_im, gamma):
        return (np.power(input_im, 1/gamma))

      original = self.img
      original = original.astype(np.float)
      original /= np.max(original)

      trans = rgb2lab(original)
      trans[:,:,0] = gc(trans[:,:,0]/100, 2.2) * 100.0
      new_img = lab2rgb(trans)*255 
      # *255 back to RGB  
      return new_img.astype(np.uint8)


    def histogram_equalization(self, a, b):

      in_im = self.img.copy()
      in_im = rgb2lab(self.img)
      channel_l = in_im[:, :, 0].astype(np.uint8)

      #l channel
      histogram_l, bin_edges_l = np.histogram(in_im[:, :, 0], bins=101)
      histogram_l = histogram_l/np.sum(histogram_l)
      cummulative_histogram_l = np.cumsum(histogram_l)

      hist_norm_im = np.zeros(self.shape)
      hist_norm_im[:, :, 0] = cummulative_histogram_l[in_im[:, :, 0].astype(int)] * 100
      hist_norm_im[:, :, 1] = in_im[:, :, 1]
      hist_norm_im[:, :, 2] = in_im[:, :, 2]


      img_new = lab2rgb(hist_norm_im)
      img_new *= 255

      return img_new.astype(np.uint8)

    # Mean and Standard Deviation
    def mean_sd(self, resize_shape):
        batch = self.batch # array of images
        arr_mean = [[],[],[]]
        arr_std = [[],[],[]]
        # numpy arrays to hold the mean and std of all images in the batch
        for item in batch:
            arr_mean[0].append(np.mean(item[0]))
            arr_mean[1].append(np.mean(item[1]))
            arr_mean[2].append(np.mean(item[2]))
            
            arr_std[0].append(np.std(item[0]))
            arr_std[1].append(np.std(item[1]))
            arr_std[2].append(np.std(item[2]))

        # calculate the mean and std of all images  
        mean = [[np.average(arr_mean[0])],[np.average(arr_mean[1])],[np.average(arr_mean[2])]]
        sd = [[np.average(arr_std[0])],[np.average(arr_std[1])],[np.average(arr_std[2])]]
        
        img_mean = np.asarray(mean)
        img_sd = np.asarray(sd)
        return (
            img_mean.astype(np.uint8),
            img_sd.astype(np.uint8)
        )

    # Batch Normalization
    def batch_norm(self, resize_shape):
        batch = self.batch
        batch_new = []
        
        # use np mean and std to calculate the bartch normalization of images
        for img in tqdm(batch):
            img = img.astype(np.float)
            in_im_norm = img/np.max(img)
            in_im_norm = resize(in_im_norm, resize_shape) #resize, cant do after mean or std has been calcaulated
            mean = np.mean(in_im_norm, axis=(0,1))
            std = np.std(in_im_norm, axis=(0,1))
            in_im_std = (in_im_norm - mean[None, None, :])/std[None, None, :]
            in_im_std_clipped = (in_im_std + 3)/6
            in_im_std_clipped = np.where(in_im_std_clipped > 1, 1, in_im_std_clipped)
            in_im_std_clipped = np.where(in_im_std_clipped < 0, 0, in_im_std_clipped)
            in_im_std_clipped = in_im_std_clipped*np.max(img)
            batch_new.append(in_im_std_clipped)

    
        batch_new = [img.astype(np.uint8) for img in batch_new]
        return batch_new
    ### Personal Implementation End ###

##### TRANSFORMATION CLASS END #####

def main():
    # Args, checks if colab or not
    if COLAB:
        global data_dir
        global output_dir
    else:
        data_dir = 'data'
        output_dir = 'output'

    # Batch input and output directories
    data_batch_dir = os.path.join(data_dir, 'batch')
    output_batch_dir = os.path.join(output_dir, 'batch')

    # Make sure data directory exists
    if not os.path.exists(data_dir):
        print(f'data_dir: {data_dir} does not exist. Exiting program...\n')
        sys.exit(1)

    # Make sure output directory exists and create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make sure output batch directory exists and create it if it doesn't
    if not os.path.exists(output_batch_dir):
        os.makedirs(output_batch_dir)

    # Get full paths for images in data directory
    img_path_list = get_image_paths(data_dir)

    # Get full paths for images in batch dir
    img_path_batch_list = get_image_paths(data_batch_dir)

    ### RUN TRANSFORMATIONS ###
    print('\nTransformations...')
    for img_path in tqdm(img_path_list):
        img_trans = transformations(img_path)

        # Flip vertically
        trans = 'flip_vertical'
        img_new = img_trans.flip_vertical()
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

        # Flip horizontally
        trans = 'flip_horizontal'
        img_new = img_trans.flip_horizontal()
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

        # Translation function
        ''' 
        Percent to shift x & y by. Positive shifts right and down,
        negative shifts left and up.
        '''
        ## Parameters you can change
        shift_x, shift_y = 0.50, 0.50
        ## Adding to output path
        trans = 'translation'
        img_new = img_trans.translation(shift_x, shift_y)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

        # Rotate function
        '''
        Rotate image by degree theta. Positive is counterclockwise,
        negative is clockwise.
        '''
        ## Paremeters you can change
        theta_degree = -45
        theta_radian = degree_to_radian(theta_degree)
        ## Adding to output path
        trans = 'rotate'
        img_new = img_trans.rotate(theta_radian)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)


        # Scaling function
        '''
        Scale the image by percent. Over 100% expands the image,
        while under 100% contracts the image. 
        '''
        ## Parameters you can change
        scale_percent = 0.50
        ## Adding to output path
        trans = 'scale'
        img_new = img_trans.scale(scale_percent)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)


        # Affine function
        '''
        User specified vector of 6 parameters for affine transformation.
        '''
        ## Parameters you can change
        a_00 = 0.8
        a_01 = -0.4
        a_10 = 0.2
        a_11 = 0.7
        t_x  = 0.0
        t_y  = 0.0
        ## Adding to output path
        A = [
            [a_00, a_01, t_x],
            [a_10, a_11, t_y]
        ]
        trans = 'affine'
        img_new = img_trans.affine(A)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)


        # Projective function
        '''
        User specified vector of 9 parameters for projective transformation.
        '''
        ## Parameters you can change
        h_00 = 1.0
        h_01 = 0.3
        h_02 = 0.0
        h_10 = 0.2
        h_11 = 0.6
        h_12 = 0.0
        h_20 = 0.0
        h_21 = 0.0
        h_22 = 0.9
        ## Adding to output path
        H = [
            [h_00, h_01, h_02],
            [h_10, h_11, h_12],
            [h_20, h_21, h_22]
        ]
        trans = 'projective'
        img_new = img_trans.projective(H)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

        # Brightness and Contrast Modulation Function
        '''
        Contrast and brightness modulation of the L channel.
        '''
        ## Parameters you can change
        a = 1.1
        b = 0.2
        ## Adding to output path
        trans = 'brightness_contrast'
        img_new = img_trans.brightness_contrast(a, b)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

        # Gamma Correction
        '''
        Gamma correction of the L channel
        '''
        ## Parameters you can change
        a = 1.1
        b = 0.2
        ## Adding to output path
        trans = 'gamme_correction'
        img_new = img_trans.gamma_correction(a, b)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

        # Histogram Equalization
        '''
        Histogram equalization for the L channel
        '''
        ## Parameters you can change
        a = 1.1
        b = 0.2
        ## Adding to output path
        trans = 'histogram_equalization'
        img_new = img_trans.histogram_equalization(a, b)
        out_path = get_output_path(output_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)

    ### NORMALIZE FUNCTIONS ###
    # Setup batch images
    img_trans = transformations(None, img_path_batch_list)

    # Compute Mean and SD Image function
    '''
    Calculates mean and sd images.
    '''

    resize_w, resize_h = 300, 300
    ## Adding to output path
    resize_shape = (resize_h, resize_w)
    img_mean, img_sd = img_trans.mean_sd(resize_shape)
    out_path1 = os.path.join(output_batch_dir, 'batch-mean.png')
    out_path2 = os.path.join(output_batch_dir, 'batch-sd.png')
    imageio.imwrite(out_path1, img_mean)
    imageio.imwrite(out_path2, img_sd)


    # Batch Normalization function
    '''
    Runs batch normalization on a batch of 10 images
    '''

    resize_w, resize_h = 300, 300

    ## Adding to output path
    print('\nBatch Normalization...')
    resize_shape = (resize_h, resize_w)
    batch = img_trans.batch_norm(resize_shape)
    trans = 'batch_norm'
    print('\nSaving batch normalized images...')
    for i in tqdm(range(len(batch))):
        img_new = batch[i]
        img_path = img_path_batch_list[i]
        out_path = get_output_path(output_batch_dir, img_path, trans)
        imageio.imwrite(out_path, img_new)
        
if __name__ == '__main__':
    main()