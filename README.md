# Jail-Breaking-Deep-models
In this project we explore effective adversarial attacks. We ex-plore launching effective adversarial attacks on production grade, publicly posted models, and degrade their performance. We have a total of five tasks to complete for this project


how many pixels are allowed to be perturbed; these are called L0 or patch attacks.)
Tasks
For your project, implement the following tasks.

Task 1: Basics
The goal is to attack a ResNet-34 model that is trained to classify the ImageNet-1K
dataset. ImageNet-1K is a well-known dataset in computer vision research with visually
challenging images from 1000 classes, and networks which are trained on ImageNet
typically also do well on other tasks. You can download the ResNet-34 model from
TorchVision using the following command:
pretrained_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
Download the attached test dataset. This is a subset of images taken from 100 classes of
the ImageNet-1K dataset. The included .json file has the associated label names and
ImageNet label indices. We will need to preprocess the images like this before doing
anything else:
mean_norms = np.array([0.485, 0.456, 0.406])
std_norms = np.array([0.229, 0.224, 0.225])
plain_transforms = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=mean_norms,
std=std_norms)
])
dataset_path = "./TestDataSet"
dataset = torchvision.datasets.ImageFolder(root=dataset_path,
transform=plain_transforms)
Evaluate the pre-trained ResNet-34 model on this test dataset; note that to validate
a prediction you will have to look at the predicted class label and match it to the
corresponding index in the .json file.
Report top-1 and top-5 accuracy for this dataset. (Top-k accuracy is calculated as
follows: compute the k most likely class labels according to the classifier, and return
True if any of these k labels matches the ground truth.)


Task 2: Pixel-wise attacks
A common and simple algorithm for mounting an L∞ attack is called Fast Gradient
Sign Method (FGSM); this implements a single step of gradient ascent (in pixel space)
and truncates the values of the gradients to at most ε. Mathematically, we can write this
as
x ← x + ε sign (∇xL)
where L is the cross-entropy loss, the gradient is with respect to the input parameters
(not the weights – so remember to ), and the sign operation just truncates the gradient to
the unit L∞ cube. (Convince yourself that this makes sense!)
The parameter ε is called the attack budget. If raw (unpreprocessed) images have pixel
values of 0-255, an attack budget of ε = 0.02 roughly corresponds to changing each
pixel value in the raw image by at most +/-1.
2
Implement FGSM for each image in the test dataset for ε = 0.02. Visualize 3 to 5 test
cases where the original model no longer classifies as expected. Your visualization can
be similar to the example shown above.
You should now have a new set of 500 images; verify that the new images are visually
similar to the original test set and that the L∞ distance between new and original is no
greater than ε = 0.02. Save this dataset (call this “Adversarial Test Set 1”). Evaluate
ResNet-34 performance and report new top-1 and top-5 accuracy scores. You should
strive to achieve accuracy drop of at least 50% relative to your baseline numbers from
Task 1 (so if your earlier metrics were above 80%, then your new metrics should be
below 30%.)

Task 3: Improved attacks
Now that you have two accuracy metrics (one for the original test set, Adversarial Test
Set 1), propose ways to improve your attack and degrade performance even further.
Remember: you can do whatever you like to the original test images, as long as the ε
constraint is met and you get worse performance than FGSM. Options include: multiple
gradient steps, targeted attacks, other optimizers, etc.
You should now have a new set of 500 images; verify that the new images are visually
similar to the original test set and that the L∞ distance between new and original is no
greater than ε = 0.02. Save this dataset (call this “Adversarial Test Set 2”). Visualize
performance for 3-5 example images. Evaluate ResNet-34 performance and report new
top-1 and top-5 accuracy scores. You should strive to achieve accuracy drop of at least
70% relative to your baseline numbers from Task 1.

Task 4: Patch attacks
Pick your best performing attack method so far, but now implement it such that you
aren’t perturbing the whole test image, but only a small random patch of size 32x32.
This is going to be more challenging, since as the attacker you have fewer knobs to
twiddle around. Therefore you are free to increase ε to a much larger amount (say 0.3
or even 0.5) to make your attack work. Hint: a targeted attack might be helpful in this
context.
You should now have a new set of 500 images. Save this dataset (call this “Adversarial
Test Set 3”). Visualize performance for 3-5 example images. Evaluate ResNet-34
performance and report new top-1 and top-5 accuracy scores.

Task 5: Transferring attacks
You now have three perturbed versions of the original test set. Evaluate classification
accuracy of these datasets using any pre-trained network other than ResNet-34. You can
choose any model you like; for example, DenseNet-121:
new_model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
A full list of ImageNet-1K models are available at this link.
3
