# Face-Keypoint-Detection
A project based on Chapter Five of [Modern Computer Vision with Pytorch]. Create a Neural Network that accurately finds facial keypoints, from the dataset [provided].
In the following image, the original image, with and without the given key points and a copy with the predicted ones can be seen. For the most part, the architecture is accurate with minimal losses.
![Screenshot_7](https://github.com/aristosp/Face-Keypoint-Detection/assets/62808962/b5b16cec-fba2-4b3d-a83e-02055d86bd8a)

Notably, when using Leaky ReLU the results were very similar to when ReLU was used, so I opted to keep ReLU as the activation function.



[Modern Computer Vision with Pytorch]: https://www.oreilly.com/library/view/modern-computer-vision/9781839213472/
[provided]: https://github.com/udacity/P1_Facial_Keypoints
