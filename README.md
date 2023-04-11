# nlcm
small IR target detection using NLCM method
Effective detection of small targets plays a pivotal
role in infrared (IR) search and track applications for modern
military defense or attack. Consequently, an effective small IR
target detection algorithm based on a novel local contrast
measure (NLCM) is proposed in here.
Initially, difference of Gaussian band-pass filter is employed 
to enhance target and suppress background clutter.
Then, a segmentation operation is implemented to obtain IR local 
regions of fixed size larger than general IR small target size.
Finally, the salient map is obtained using the NLCM, and an 
adaptive threshold is applied to extract the target region.
