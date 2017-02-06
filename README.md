# CarND-BehavioralCloning-Solution
Project 3 of the Udacity Self-Driving Car Degree: Behavioral Cloning


This is my best model running on the first track at 0.45 throttle. I am using a modified NVidia steering model, HSV space, and only use image flips, large horizontal shifts, and brightness adjustments when generating augmented data for the model. I am training on the first track data only: actually on a very small fraction of Udacity data.  I only train for one epoch on 10240 generated images! 

[![Car driving on the first track](http://img.youtube.com/vi/Y7p7gA194qg/0.jpg)](https://youtu.be/Y7p7gA194qg)

This is my model generalizing to the second track. The car runs at 0.45 throttle and completes the track in 2 minutes 31.77 seconds! 

[![Car driving on the second track](http://img.youtube.com/vi/wUyuApyLNk0/0.jpg)](https://youtu.be/wUyuApyLNk0)
