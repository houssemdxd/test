# smart security system

This project is about a smart security system that helps to protect farmer animals and products from wild pigs attaque.
so ,I trained a model to detect wild pigs , if the camera  detected a threat  the system would run a very loud sound in order to keep the animal away from the farm  .
I use the tensorflow API.and for this project its possible to work with a camera but I choose to test on a video .


<h3>Run it in your local machine :</h3>
- You have to install tensorflow 1.15 dependencies.(virtual environment is recommended)<br>
- you have also to find  the configuration  file
- go to \models\models-master\research\object_detection\training\ssd_mobilenet_v1_pets.config"
- change the 156 line  in the file :<br>
- fine_tune_checkpoint:"{put where you save the project path}\\models\\models-master\\research\\object_detection\\ssd_mobilenet_v1_coco_2018_01_28\\ssd_mobilenet_v1_coco_2018_01_28\\model.ckpt"
   
- Go to models\models-master\research\object_detection<br>
- Open cmd <br>
- Type python vid.py <br>
    you will see an example of a system running of video (there is more videos you can test all you will found in object detection folder )<br>
<h2>please watch the complete video( (/test/detection.mp4) to see more example !!!!</h2><br>
<h2>Or just watch this video to take a quick recap</h2>

<h3>with detection <h3>   

https://user-images.githubusercontent.com/29411920/208974280-a3c18bb9-28b4-4979-8918-e653f1fdb90f.mp4

<h3>without detection <h3>   


https://user-images.githubusercontent.com/29411920/208938473-5f9126ea-a740-4460-9d51-0f3c6380e9b5.mp4

