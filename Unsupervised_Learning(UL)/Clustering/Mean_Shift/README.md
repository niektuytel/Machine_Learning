# Mean Shift
[`Mean Shift`](https://www.youtube.com/watch?v=Evc53OaDTFc) clustering is a sliding-window-based algorithm that attempts to find dense areas of data points.  
It is a centroid-based algorithm meaning that the goal is to locate the center points of each group/class, 
which works by updating candidates for center points to be the mean of the points within the sliding-window.  
These candidate windows are then filtered in a post-processing stage to eliminate near-duplicates, 
forming the final set of center points and their corresponding groups


<p align="center">
  <img src="../../../_EXTRA/images/ml_clustering_mean_shift_0.png">
</p>
<p align="center">
  <img src="https://github.com/mattnedrich/MeanShift_py/raw/master/sample_images/ms_3d_image_animation.gif">
</p>
<p align="center">
    <img src="https://github.com/mattnedrich/MeanShift_py/raw/master/sample_images/mean_shift_image.jpg" width=300>
    -> result ->
    <img src="https://github.com/mattnedrich/MeanShift_py/raw/master/sample_images/mean_shift_image_clustered.png" width=300>
</p>


### code 
[`python3 mean_shift_scratch.py`](./mean_shift_scratch.py)  
[`python3 mean_shift.py`](./mean_shift.py)  

## Resources
http://primo.ai/index.php?title=Mean-Shift_Clustering  
https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/  
https://towardsdatascience.com/speeding-up-your-code-1-the-example-of-the-mean-shift-clustering-in-poincar%C3%A9-ball-space-d46169bfdfc8  
https://github.com/zziz/mean-shift
