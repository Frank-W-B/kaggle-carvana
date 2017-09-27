Pan of attack for Carvana challenge
************************************

* Verify that the image-segmentation-keras works on the provided files.
* Determine a minimal architecture that trains faster locally.
* Switch carvana images and masks to input needed for the  
  image-segmentation-keras script.
* Get some preliminary results.
* Merged code suggestions image-segmentation-keras and keras-segnet to
  get working code.  Predicts with 94% accuracy on bw images in keras-
  segnet.
* Code works locally and with AWS, AWS 16%-73% faster depending on
  model complexity.  AWS Tesla K80. 4992 NVIDIA cores. I have GeForce
  GTX 960M with 640 cores.
* Now need to get Carvana images to a comparable size.  Look at images
  and decide on size.
* Carvana images are 1280 x 1918.  If I crop to 1272 x 1908 I can use
  a square window 212 x 212 to divide image into 6 windows high and
  9 windows wide - so 54 images per image, double to 108 with mirroring.
  Roughly size of VGG images - later SegNet was 360 x 480.  There are
  5088 training images so that's 54 * 5088 =  274752 no mirroring, or
  108 * 5088 = 549504 images with mirroring.  With 5088/16 = 318 cars in dataset,
  can train on 256 (4096), test on 31 (496), hold out on 31 (496).
  4096 * 108 = 442,368 train
  496 * 108 = 53,568 test
  496 * 108 = 53,568 hold-out
* Need to calculate Dice coefficient = TP/(P + FP)
  TP is sum of element-wise multiplication of prediction and ground truth
  P is sum of all positive in ground truth
  PP is the sum of all predicticted positive in image
  FP = PP - TP
* Make a list of cars, and make a list of cars in train, test, and holdout
* So read in image (skimage), crop_outside, then crop into 54 images, flip them
  then save into array with that car's name

 
