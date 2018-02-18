# ai-face-lock-jetson

This is a guide on how to train AI for facial recongition, and deploy it onto Nvidia Jetson to unlock deadbolts, turn on LEDs, tracking in and outs, and many other functionality.  We use AI to detect the facial recongition, and we use Walabot Radar to detect the energy level around so that it won't be faked.

# Training 
We used this project https://github.com/hqli/face_recognition to help training our data.  Taking about 10,000 images of myself and 10,000 images of everyone else to make it happen.

# IoT Part
The Walabot is going to be build by end of Feburary to ensure the security of faking images.
