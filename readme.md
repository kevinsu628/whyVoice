# whyVoice
## Introduction

Smart assistants, such as Google Assistant, Siri, and Alexa, have been deeply integrated into our lives. 
With one simple command, people can wake up their smart assistant and ask them to do them a favor ranging from 
setting up a timer to turning on the light. However, built on voice commands, current smart assistants do have some
limitations. One of them is that mute people cannot use smart assistants unless by manually opening the app and type 
the command. When it's not very convenient to speak, communicating with a smart assistant is also not a very good 
experience. That's why we built whyVoice, an idea of using smart assistants without the need to say anything.

Our system is expected to recognize a key rhythm (such as the sound of knocking the desk with a specific rhythm), 
record the hand gesture movement using its camera, and output the classification of the hand gesture video 
(such as turn on the light). The whole experience does not need the user to speak a word.

## Usage
[YouTube demo](https://www.youtube.com/watch?v=LbEpr_3XYmM)

### Create the conda environment

<pre><code>git clone https://github.com/kevinsu628/whyVoice.git
conda env create -f environment.yml
</code></pre>

### Download hand gesture recognition model 
Download using [this link](https://drive.google.com/file/d/1fNP-fXooKWjvOV_IyvkXy0MA3z8WQTaF/view?usp=sharing) 

<pre><code>## relocate the hand gesture model
mv /path_to_model src/hgr/model
</code></pre>

### Build
#### Option 1: If you would like to build on ROS:
Our system is currently running on (Robotics Operating System) ROS on a Ubuntu 20.4. 
Although achieving our idea doesn't strictly require ROS (i.e. you can make them work
in a jupyter notebook), we did it so that it can be used in a real hardware one day.

1. Follow [this link](http://wiki.ros.org/noetic/Installation/Ubuntu) to install ROS Noetic 
on Ubuntu 20.04.
2. Modify model path (absolute path suggested) in 
   [KWS node](https://github.com/kevinsu628/whyVoice/blob/97a262bde135d7cbddd6f1c45efb4ab3360a724b/src/kws/scripts/kws_node.py#L124) 
   and 
   [HGR node](https://github.com/kevinsu628/whyVoice/blob/97a262bde135d7cbddd6f1c45efb4ab3360a724b/src/hgr/scripts/hgr_node.py#L100)
3. Launch the program
<pre><code>## in terminal 1 to run roscore: 
roscore

## in terminal 2 to run keyword spotting node and hand gesture recognition node
catkin_make
source devel/setup.bash
roslaunch whyvoice.launch

## Optional: To view the real time keyword spotting output in terminal 3:
rostopic echo /kws
</code></pre>
**Note**: 
1. Remember to run "source /opt/ros/noetic/setup.bash" every time you start a terminal
to activate the ROS environment.
   
2. To make python in ROS and python in Anaconda work together is tricky. You need to set up
PYTHONPATH so that packages of both conda and ROS can be found. If you have any error related to python path, 
try something like this command:
<pre><code>export PYTHONPATH=/home/username/anaconda3/envs/whyvoice/lib/python3.6/site-packages:
$PYTHONPATH:/usr/lib/python3/dist-packages/</code></pre>

Follow the rhythm used in the YouTube demo linked above to activate the system (knock the desk 4 times). The system 
should wake up the camera and start hand gesture recognition. Try the following 4 gestures:
1. swipe to right
2. swipe to left
3. hand zoom in
4. hand zoom out



#### Option 2: If you would like to run pure Python on any system:
Stay tuned for the non-ROS version of our this project. I will create a new branch later.
